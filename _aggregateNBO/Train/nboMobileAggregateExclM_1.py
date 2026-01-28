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
# MAGIC ###### Mobile

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, TimestampType
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Paramètres
container = "next-best-offer"
storage_account = "blobstoragehandson"
sub_folder = "raw_data/mobile/*/mobile.parquet"

endpoint = f"wasbs://{container}@{storage_account}.blob.core.windows.net/{sub_folder}"

# Définition du schéma
mobile_schema = StructType([
    StructField("CUST_NUM", LongType(), True),
    StructField("EVT_DT", TimestampType(), True),
    StructField("OUT_OF_BUNDLE", LongType(), True)
])

# Lire tous les fichiers parquet
df_mobile = spark.read.schema(mobile_schema).parquet(endpoint)

# Identifier la date max réelle
max_date = df_mobile.agg(F.max("EVT_DT")).collect()[0][0]
print("Dernière date disponible :", max_date)

# Calculer les deux derniers mois à exclure
last_month_date = datetime(max_date.year, max_date.month, 1)
exclude_months = [
    last_month_date,  # dernier mois
    last_month_date - relativedelta(months=1)  # mois précédent
]
exclude_year_month = [(d.year, d.month) for d in exclude_months]

# Ajouter colonnes année et mois TEMPORAIRES pour le filtrage
df_mobile = df_mobile.withColumn("_year_tmp", F.year("EVT_DT")) \
                  .withColumn("_month_tmp", F.month("EVT_DT"))

# Filtrer pour exclure les deux derniers mois
condition = ~(
    ((F.col("_year_tmp") == exclude_year_month[0][0]) & (F.col("_month_tmp") == exclude_year_month[0][1])) |
    ((F.col("_year_tmp") == exclude_year_month[1][0]) & (F.col("_month_tmp") == exclude_year_month[1][1]))
)
df_mobile = df_mobile.filter(condition)

# Supprimer les colonnes temporaires
df_mobile = df_mobile.drop("_year_tmp", "_month_tmp")

# COMMAND ----------

df_mobile.dropDuplicates()

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

df_mobile   = preprocess_df(df_mobile)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Mobile

# COMMAND ----------

from pyspark.sql import functions as F
from dateutil.relativedelta import relativedelta

# Obtenir la date maximale (référence)
date_max_mobile = df_mobile.agg(F.max("EVT_DT")).first()[0]

# Générer la liste des 3 derniers mois (mois, année)
month_year_l3m_mobile = [((date_max_mobile - relativedelta(months=i)).month,
                          (date_max_mobile - relativedelta(months=i)).year) for i in range(3)]

# Filtrer sur ces 3 mois
df_mobile = df_mobile.filter(
    F.struct("MONTH", "YEAR").isin([F.struct(F.lit(m), F.lit(y)) for m, y in month_year_l3m_mobile])
)

# Agréger par client/mois/année (premier niveau d'agrégation)
df_mobile = df_mobile.groupBy("CUST_NUM", "MONTH", "YEAR").agg(
    F.sum("OUT_OF_BUNDLE").alias("AMOUNT_OUT_OF_BUNDLE"),
    F.countDistinct("EVT_DT").alias("NB_DAYS_OUT_OF_BUNDLE")
)

# Réagréger sur les 3 derniers mois : total par client et garder le mois/année de référence
df_mobile = df_mobile.groupBy("CUST_NUM").agg(
    F.sum("AMOUNT_OUT_OF_BUNDLE").alias("AMOUNT_OUT_OF_BUNDLE_L3M"),
    F.sum("NB_DAYS_OUT_OF_BUNDLE").alias("NB_DAYS_OUT_OF_BUNDLE_L3M"),
    F.max("MONTH").alias("MONTH"),
    F.max("YEAR").alias("YEAR")
)

df_mobile = df_mobile.drop("EVT_DT")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# Conversion Pandas -> Spark pour sauvegarde Delta
#df_final = spark.createDataFrame(df_final)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboMobileAggregateExclM_1;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_mobile.write.format("delta").saveAsTable("nboMobileAggregateExclM_1")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboMobileAggregateExclM_1;
