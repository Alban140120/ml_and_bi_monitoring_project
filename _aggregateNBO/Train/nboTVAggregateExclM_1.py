# Databricks notebook source
# MAGIC %md
# MAGIC ### Connexion au blob storage

# COMMAND ----------

# Connexion au blob storage
spark.conf.set("fs.azure.account.key.blobstoragehandson.blob.core.windows.net", "x")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import des parquets dans un dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ###### TV

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Paramètres
container = "next-best-offer"
storage_account = "blobstoragehandson"
sub_folder = "raw_data/tv/*/tv.parquet"

endpoint = f"wasbs://{container}@{storage_account}.blob.core.windows.net/{sub_folder}"

# Définition du schéma
tv_schema = StructType([
    StructField("CUST_NUM", LongType(), True),
    StructField("EVT_DT", TimestampType(), True),
    StructField("TV_PROGRAM", StringType(), True),
    StructField("DURATION", LongType(), True)
])

# Lire tous les fichiers parquet
df_tv = spark.read.schema(tv_schema).parquet(endpoint)

# Identifier la date max réelle
max_date = df_tv.agg(F.max("EVT_DT")).collect()[0][0]
print("Dernière date disponible :", max_date)

# Calculer les deux derniers mois à exclure
last_month_date = datetime(max_date.year, max_date.month, 1)
exclude_months = [
    last_month_date,  # dernier mois
    last_month_date - relativedelta(months=1)  # mois précédent
]
exclude_year_month = [(d.year, d.month) for d in exclude_months]

# Ajouter colonnes année et mois TEMPORAIRES pour le filtrage
df_tv = df_tv.withColumn("_year_tmp", F.year("EVT_DT")) \
              .withColumn("_month_tmp", F.month("EVT_DT"))

# Filtrer pour exclure les deux derniers mois
condition = ~(
    ((F.col("_year_tmp") == exclude_year_month[0][0]) & (F.col("_month_tmp") == exclude_year_month[0][1])) |
    ((F.col("_year_tmp") == exclude_year_month[1][0]) & (F.col("_month_tmp") == exclude_year_month[1][1]))
)
df_tv = df_tv.filter(condition)

# Supprimer les colonnes temporaires
df_tv = df_tv.drop("_year_tmp", "_month_tmp")

# COMMAND ----------

df_tv.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import to_date, month, year

def preprocess_df(df: DataFrame, duration_col: str = None) -> DataFrame:
    """
    Standardise un DataFrame pour ton pipeline ML :
    - Convertit la colonne EVT_DT en Date
    - Crée les colonnes MONTH et YEAR
    - Renomme la colonne DURATION si spécifiée
    - Supprime systématiquement EVT_DT après création des colonnes MONTH/YEAR

    Args:
        df (DataFrame): DataFrame Spark
        duration_col (str, optional): nom pour renommer la colonne DURATION

    Returns:
        DataFrame: DataFrame traité
    """
    # Convertir EVT_DT en date
    df = df.withColumn("EVT_DT", to_date("EVT_DT"))
    
    # Renommer la colonne DURATION si nécessaire
    if duration_col:
        df = df.withColumnRenamed("DURATION", duration_col)
    
    # Créer MONTH et YEAR
    df = df.withColumn("MONTH", month("EVT_DT")) \
           .withColumn("YEAR", year("EVT_DT")) #\
           #.drop("EVT_DT")
    
    return df

# COMMAND ----------

df_tv       = preprocess_df(df_tv, duration_col="DURATION_TV")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### TV

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, lit, struct, max as F_max, countDistinct, percent_rank
from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame

def process_l3m_tv(df_tv: DataFrame, n_months: int = 3) -> DataFrame:
    """
    Calcul L3M pour df_tv : durée, jours distincts, rank, pivot et renommage.
    
    Args:
        df_tv (DataFrame): DataFrame TV avec EVT_DT et DURATION_TV
        n_months (int): nombre de mois pour L3M
    
    Returns:
        DataFrame: pivoté, rang percentile calculé et colonnes renommées
    """
    
    # Date max et période L3M
    date_max = df_tv.agg(F.max("EVT_DT")).first()[0]
    month_year_l3m = [
        ((date_max - relativedelta(months=i)).month, (date_max - relativedelta(months=i)).year)
        for i in range(n_months)
    ]
    df_tv = df_tv.filter(
        struct("MONTH", "YEAR").isin([struct(lit(m), lit(y)) for m, y in month_year_l3m])
    )

    # Agrégation durée + jours distincts
    df_agg = df_tv.groupBy("CUST_NUM", "TV_PROGRAM", "MONTH", "YEAR").agg(
        F.sum("DURATION_TV").alias("TV_DURATION_L3M"),
        F.countDistinct("EVT_DT").alias("TV_NB_DAYS_USAGE_L3M")
    )

    # Rang percentile par programme
    window_spec = Window.partitionBy("TV_PROGRAM", "MONTH", "YEAR").orderBy("TV_DURATION_L3M")
    df_agg = df_agg.withColumn("PERCENT_RANK", percent_rank().over(window_spec)) \
                   .withColumn("TV_RANK", (col("PERCENT_RANK") * 99 + 1).cast("int")) \
                   .drop("PERCENT_RANK")

    # Pivot
    df_pivoted = df_agg.groupBy("CUST_NUM", "MONTH", "YEAR").pivot("TV_PROGRAM").agg(
        F_max("TV_DURATION_L3M"),
        F_max("TV_NB_DAYS_USAGE_L3M"),
        F_max("TV_RANK")
    )

    # Renommage dynamique des colonnes
    renamed_columns = []
    for c in df_pivoted.columns:
        if c in ["CUST_NUM", "MONTH", "YEAR"]:
            continue
        if "_max(TV_DURATION_L3M)" in c:
            cat = c.replace("_max(TV_DURATION_L3M)", "")
            new_col = f"TV_DURATION_{cat}_L3M_SECONDES"
        elif "_max(TV_NB_DAYS_USAGE_L3M)" in c:
            cat = c.replace("_max(TV_NB_DAYS_USAGE_L3M)", "")
            new_col = f"TV_DURATION_{cat}_L3M_DAYS"
        elif "_max(TV_RANK)" in c:
            cat = c.replace("_max(TV_RANK)", "")
            new_col = f"TV_RANK_{cat}_L3M"
        else:
            new_col = c
        renamed_columns.append((c, new_col))

    for old_col, new_col in renamed_columns:
        df_pivoted = df_pivoted.withColumnRenamed(old_col, new_col)

    # Supprimer EVT_DT car plus utile après calcul
    df_result = df_pivoted.drop("EVT_DT")

    return df_result

# COMMAND ----------

df_tv = process_l3m_tv(df_tv)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboTVAggregateExclM_1;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_tv.write.format("delta").saveAsTable("nboTVAggregateExclM_1")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboTVAggregateExclM_1;
