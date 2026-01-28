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
# MAGIC ###### Socio demo

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, TimestampType
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Paramètres
container = "next-best-offer"
storage_account = "blobstoragehandson"
sub_folder = "raw_data/socio_demo/*/socio_demo.parquet"

endpoint = f"wasbs://{container}@{storage_account}.blob.core.windows.net/{sub_folder}"

# Définition du schéma
socio_demo_schema = StructType([
    StructField("CUST_NUM", LongType(), True),
    StructField("EVT_DT", TimestampType(), True),
    StructField("CUST_AGE", LongType(), True),
    StructField("NB_MOBILE_SUBS", LongType(), True),
    StructField("NB_DAYS_INTERNET_LINE", LongType(), True),
    StructField("NB_DAYS_TV_LINE", LongType(), True)
])

# Lire tous les fichiers parquet
df_socio_demo = spark.read.schema(socio_demo_schema).parquet(endpoint)

# Identifier la date max réelle
max_date = df_socio_demo.agg(F.max("EVT_DT")).collect()[0][0]
print("Dernière date disponible :", max_date)

# Calculer les deux derniers mois à exclure
last_month_date = datetime(max_date.year, max_date.month, 1)
exclude_months = [
    last_month_date,  # dernier mois
    last_month_date - relativedelta(months=1)  # mois précédent
]
exclude_year_month = [(d.year, d.month) for d in exclude_months]

# Ajouter colonnes année et mois TEMPORAIRES pour le filtrage
df_socio_demo = df_socio_demo.withColumn("_year_tmp", F.year("EVT_DT")) \
                      .withColumn("_month_tmp", F.month("EVT_DT"))

# Filtrer pour exclure les deux derniers mois
condition = ~(
    ((F.col("_year_tmp") == exclude_year_month[0][0]) & (F.col("_month_tmp") == exclude_year_month[0][1])) |
    ((F.col("_year_tmp") == exclude_year_month[1][0]) & (F.col("_month_tmp") == exclude_year_month[1][1]))
)
df_socio_demo = df_socio_demo.filter(condition)

# Supprimer les colonnes temporaires
df_socio_demo = df_socio_demo.drop("_year_tmp", "_month_tmp")

# COMMAND ----------

df_socio_demo.dropDuplicates()

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

df_socio_demo = preprocess_df(df_socio_demo)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Socio démographique

# COMMAND ----------

from pyspark.sql import functions as F

# Agrégation par CUST_NUM, MONTH et YEAR pour les moyennes
df_socio_demo = df_socio_demo.groupBy("CUST_NUM", "MONTH", "YEAR").agg(
    F.avg("CUST_AGE").alias("AVG_CUST_AGE"),
    F.avg("NB_MOBILE_SUBS").alias("AVG_NB_MOBILE_SUBS"),
    F.avg("NB_DAYS_INTERNET_LINE").alias("AVG_NB_DAYS_INTERNET_LINE"),
    F.avg("NB_DAYS_TV_LINE").alias("AVG_NB_DAYS_TV_LINE")
)

df_socio_demo = df_socio_demo.drop("EVT_DT")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboSocioDemoAggregateExclM_1;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_socio_demo.write.format("delta").saveAsTable("nboSocioDemoAggregateExclM_1")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboSocioDemoAggregateExclM_1;
