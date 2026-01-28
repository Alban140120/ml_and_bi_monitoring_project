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
# MAGIC ###### Internet

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Paramètres
container = "next-best-offer"
storage_account = "blobstoragehandson"
sub_folder = "raw_data/internet/*/internet.parquet"

endpoint = f"wasbs://{container}@{storage_account}.blob.core.windows.net/{sub_folder}"

# Définition du schéma
internet_schema = StructType([
    StructField("CUST_NUM", LongType(), True),
    StructField("EVT_DT", TimestampType(), True),
    StructField("WEBSITE_TYPE", StringType(), True),
    StructField("DURATION", LongType(), True)
])

# Lire tous les fichiers parquet
df_internet = spark.read.schema(internet_schema).parquet(endpoint)

# Identifier la date max réelle
max_date = df_internet.agg(F.max("EVT_DT")).collect()[0][0]
print("Dernière date disponible :", max_date)

# Calculer les deux derniers mois à exclure
last_month_date = datetime(max_date.year, max_date.month, 1)
exclude_months = [
    last_month_date,  # dernier mois
    last_month_date - relativedelta(months=1)  # mois précédent
]
exclude_year_month = [(d.year, d.month) for d in exclude_months]

# Ajouter colonnes année et mois TEMPORAIRES pour le filtrage
df_internet = df_internet.withColumn("_year_tmp", F.year("EVT_DT")) \
                    .withColumn("_month_tmp", F.month("EVT_DT"))

# Filtrer pour exclure les deux derniers mois
condition = ~(
    ((F.col("_year_tmp") == exclude_year_month[0][0]) & (F.col("_month_tmp") == exclude_year_month[0][1])) |
    ((F.col("_year_tmp") == exclude_year_month[1][0]) & (F.col("_month_tmp") == exclude_year_month[1][1]))
)
df_internet = df_internet.filter(condition)

# Supprimer les colonnes temporaires
df_internet = df_internet.drop("_year_tmp", "_month_tmp")

# COMMAND ----------

df_internet.dropDuplicates()

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

df_internet = preprocess_df(df_internet, duration_col="DURATION_INTERNET")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Internet

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, lit, struct, max as F_max, countDistinct, percent_rank
from pyspark.sql import DataFrame
from dateutil.relativedelta import relativedelta

def process_l3m_internet(df_internet: DataFrame, n_months: int = 3) -> DataFrame:
    """
    Calcule les métriques L3M par CUST_NUM et WEBSITE_TYPE avec pivot, rank et noms de colonnes standardisés.
    
    Args:
        df_internet (DataFrame): DataFrame Internet avec EVT_DT et DURATION_INTERNET
        n_months (int): nombre de mois à prendre en compte pour L3M (default=3)
    
    Returns:
        DataFrame: métriques pivotées et renommées
    """

    # Date max
    date_max = df_internet.agg(F.max("EVT_DT")).first()[0]

    # Liste des derniers n_months
    month_year_l3m = [
        ((date_max - relativedelta(months=i)).month, (date_max - relativedelta(months=i)).year)
        for i in range(n_months)
    ]

    # Filtrer sur ces mois
    df_internet = df_internet.filter(
        struct("MONTH", "YEAR").isin([struct(lit(m), lit(y)) for m, y in month_year_l3m])
    )

    # Agrégation
    df_agg = df_internet.groupBy("CUST_NUM", "WEBSITE_TYPE", "MONTH", "YEAR").agg(
        F.sum("DURATION_INTERNET").alias("DURATION_L3M"),
        F.countDistinct("EVT_DT").alias("NB_DAYS_USAGE_L3M")
    )

    # Rank par catégorie
    window_spec = Window.partitionBy("WEBSITE_TYPE", "MONTH", "YEAR").orderBy("DURATION_L3M")
    df_agg = df_agg.withColumn("PERCENT_RANK", percent_rank().over(window_spec)) \
                   .withColumn("INTERNET_RANK", (col("PERCENT_RANK") * 99 + 1).cast("int")) \
                   .drop("PERCENT_RANK")

    # Pivot
    df_pivot = df_agg.groupBy("CUST_NUM", "MONTH", "YEAR").pivot("WEBSITE_TYPE").agg(
        F_max("DURATION_L3M"),
        F_max("NB_DAYS_USAGE_L3M"),
        F_max("INTERNET_RANK")
    )

    # Renommer colonnes pour correspondre à ton ancien standard
    renamed_columns = []
    for c in df_pivot.columns:
        if c in ["CUST_NUM", "MONTH", "YEAR"]:
            continue
        if "_max(DURATION_L3M)" in c:
            cat = c.replace("_max(DURATION_L3M)", "")
            new_col = f"INTERNET_DURATION_{cat}_L3M_SECONDES"
        elif "_max(NB_DAYS_USAGE_L3M)" in c:
            cat = c.replace("_max(NB_DAYS_USAGE_L3M)", "")
            new_col = f"INTERNET_DURATION_{cat}_L3M_DAYS"
        elif "_max(INTERNET_RANK)" in c:
            cat = c.replace("_max(INTERNET_RANK)", "")
            new_col = f"INTERNET_RANK_{cat}_L3M"
        else:
            new_col = c
        renamed_columns.append((c, new_col))

    for old_col, new_col in renamed_columns:
        df_pivot = df_pivot.withColumnRenamed(old_col, new_col)

    return df_pivot

# COMMAND ----------

df_internet = process_l3m_internet(df_internet)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboInternetAggregateExclM_1;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_internet.write.format("delta").saveAsTable("nboInternetAggregateExclM_1")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboInternetAggregateExclM_1;
