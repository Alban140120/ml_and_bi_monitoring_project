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
# MAGIC ###### Journeys

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Paramètres
container = "next-best-offer"
storage_account = "blobstoragehandson"
sub_folder = "raw_data/journeys/*/journeys.parquet"

endpoint = f"wasbs://{container}@{storage_account}.blob.core.windows.net/{sub_folder}"

# Définition du schéma
journeys_schema = StructType([
    StructField("CUST_NUM", LongType(), True),
    StructField("EVT_DT", TimestampType(), True),
    StructField("CONTACT_TYPE", StringType(), True),
    StructField("NB_CONTACTS", LongType(), True)
])

# Lire tous les fichiers parquet
df_journeys = spark.read.schema(journeys_schema).parquet(endpoint)

# Identifier la date max réelle
max_date = df_journeys.agg(F.max("EVT_DT")).collect()[0][0]
print("Dernière date disponible :", max_date)

# Calculer les deux derniers mois à exclure
last_month_date = datetime(max_date.year, max_date.month, 1)
exclude_months = [
    last_month_date,  # dernier mois
    last_month_date - relativedelta(months=1)  # mois précédent
]
exclude_year_month = [(d.year, d.month) for d in exclude_months]

# Ajouter colonnes année et mois TEMPORAIRES pour le filtrage
df_journeys = df_journeys.withColumn("_year_tmp", F.year("EVT_DT")) \
                    .withColumn("_month_tmp", F.month("EVT_DT"))

# Filtrer pour exclure les deux derniers mois
condition = ~(
    ((F.col("_year_tmp") == exclude_year_month[0][0]) & (F.col("_month_tmp") == exclude_year_month[0][1])) |
    ((F.col("_year_tmp") == exclude_year_month[1][0]) & (F.col("_month_tmp") == exclude_year_month[1][1]))
)
df_journeys = df_journeys.filter(condition)

# Supprimer les colonnes temporaires
df_journeys = df_journeys.drop("_year_tmp", "_month_tmp")

# COMMAND ----------

df_journeys.dropDuplicates()

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

df_journeys = preprocess_df(df_journeys)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Journeys

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, struct, max as F_max, countDistinct
from pyspark.sql import DataFrame
from dateutil.relativedelta import relativedelta

def process_l3m_journeys(df_journeys: DataFrame, n_months: int = 3) -> DataFrame:
    """
    Calcul L3M des journeys par CUST_NUM et CONTACT_TYPE avec pivot, agrégation globale et renommage.
    
    Args:
        df_journeys (DataFrame): DataFrame journeys avec EVT_DT et NB_CONTACTS
        n_months (int): nombre de mois à prendre en compte pour L3M (default=3)
    
    Returns:
        DataFrame: métriques pivotées et renommées
    """

    # Date max
    date_max = df_journeys.agg(F.max("EVT_DT")).first()[0]

    # Liste des derniers n_months
    month_year_l3m = [
        ((date_max - relativedelta(months=i)).month, (date_max - relativedelta(months=i)).year)
        for i in range(n_months)
    ]

    # Filtrer sur ces mois
    df_journeys = df_journeys.filter(
        struct("MONTH", "YEAR").isin([struct(lit(m), lit(y)) for m, y in month_year_l3m])
    )

    # Agrégation par CONTACT_TYPE
    df_detail = df_journeys.groupBy("CUST_NUM", "CONTACT_TYPE", "MONTH", "YEAR").agg(
        F.sum("NB_CONTACTS").alias("AMOUNT_OF_CONTACT_L3M"),
        F.countDistinct("EVT_DT").alias("NB_DAYS_WITH_L3M")
    )

    # Pivot des mesures par type de contact
    df_pivot = df_detail.groupBy("CUST_NUM", "MONTH", "YEAR").pivot("CONTACT_TYPE").agg(
        F_max("AMOUNT_OF_CONTACT_L3M"),
        F_max("NB_DAYS_WITH_L3M")
    )

    # Renommage des colonnes pour correspondre à ton ancien standard
    renamed_columns = []
    for c in df_pivot.columns:
        if c in ["CUST_NUM", "MONTH", "YEAR"]:
            continue
        if "_max(AMOUNT_OF_CONTACT_L3M)" in c:
            type_ = c.replace("_max(AMOUNT_OF_CONTACT_L3M)", "")
            new_col = f"AMOUNT_OF_CONTACT_{type_}_L3M"
        elif "_max(NB_DAYS_WITH_L3M)" in c:
            type_ = c.replace("_max(NB_DAYS_WITH_L3M)", "")
            new_col = f"JOURNEY_DURATION_{type_}_L3M_DAYS"
        else:
            new_col = c
        renamed_columns.append((c, new_col))

    for old_col, new_col in renamed_columns:
        df_pivot = df_pivot.withColumnRenamed(old_col, new_col)

    # Agrégation globale
    df_global = df_journeys.groupBy("CUST_NUM", "MONTH", "YEAR").agg(
        countDistinct("EVT_DT").alias("TOTAL_NUMBER_OF_JOURNEYS_L3M"),
        F.sum("NB_CONTACTS").alias("TOTAL_NUMBER_OF_CONTACTS_L3M")
    )

    # Fusion pivot + global
    df_result = df_pivot.join(df_global, on=["CUST_NUM", "MONTH", "YEAR"], how="left")

    # Supprimer EVT_DT car plus utile
    df_result = df_result.drop("EVT_DT")

    return df_result

# COMMAND ----------

df_journeys = process_l3m_journeys(df_journeys)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboJourneysAggregateExclM_1;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_journeys.write.format("delta").saveAsTable("nboJourneysAggregateExclM_1")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboJourneysAggregateExclM_1;
