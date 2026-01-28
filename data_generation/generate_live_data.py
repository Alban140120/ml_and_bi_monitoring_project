# Databricks notebook source
# MAGIC %md
# MAGIC Install libraries - you can also directly install the libraries on your cluster and avoid running this cell

# COMMAND ----------

# MAGIC %pip install numpy==1.22
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install fsspec
# MAGIC %pip install pyyaml
# MAGIC %pip install databricks-sdk
# MAGIC %pip install adlfs

# COMMAND ----------

# MAGIC %md
# MAGIC Import librairies - import the librairy used to generate the customer data

# COMMAND ----------

import sys
from utils import generate_data
sys.path.append('../utils')
from external_config import parse_config
from read_write import read_config
import databricks
from databricks.sdk.runtime import *
from azure.storage.blob import BlobServiceClient

from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC Import the config for customer data generation

# COMMAND ----------

config = read_config("config/config_generation.yml")
global_config, target_definition_config, population_distribution_config, storage_config = parse_config(config)

# COMMAND ----------

# MAGIC %md
# MAGIC Generate and save data in the relevant folder in blob storage (set in the above config file)

# COMMAND ----------

folder_name = 'live'

current_date = datetime.now().date()
start_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
end_date = start_date

generate_data.generate_datasets(start_date=start_date, end_date=end_date, folder_name=folder_name,
                                population_distribution_config=population_distribution_config, target_definition_config=target_definition_config, storage_config=storage_config, global_config=global_config)
