"""
Generic file for automatic dataset generation
"""

import pandas as pd
import adlfs
import numpy as np
from datetime import datetime, timedelta, date
import os
import logging

from utils.external_config import parse_config
from utils.read_write import read_config
import databricks
from databricks.sdk.runtime import *
from azure.storage.blob import BlobServiceClient

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import numpy as np
import pandas as p
from utils.read_write import save_to_blob



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

Logger = logging.getLogger(__name__)




def generate_noisy_data(configs, std_multiplier, num_rows_per_dataset):
    df_list = []
    # Extract keys from the dictionary
    config_keys = list(configs.keys())

    num_configs = len(config_keys)

    for i in range(num_configs):
        for j in range(i + 1, num_configs):
            # Get configurations for each pair from the dictionary
            config_i = configs[config_keys[i]]
            config_j = configs[config_keys[j]]

            # Compute the average configuration for each pair
            avg_config = average_noisy_configs([config_i, config_j], std_multiplier=std_multiplier)

            # Generate dataset using the average configuration
            df = generate_data(avg_config, num_rows=num_rows_per_dataset)

            # Append the generated dataset to the list
            df_list.append(df)

    # Concatenate all the datasets
    final_df = pd.concat(df_list, axis=0, ignore_index=True)

    return final_df


def generate_data(config, num_rows):
    # Generate a random number of rows from a normal distribution with mean=num_rows
    actual_num_rows = int(np.random.normal(num_rows, num_rows/10)) 

    # Create a list to store data for each column
    data = []

    # create CUST_NUM values
    cust_num_values = np.random.randint(50000, 80000000, actual_num_rows)
    data.append(cust_num_values)

    # Generate data for each column using numpy
    for col, params in config.items():
        mean = params['mean']
        std = params['std']
        clip = params['clip']

        # Generate random data based on mean and std
        values = np.random.normal(mean, std, actual_num_rows)

        # Clip values if necessary
        values = np.clip(values, a_min=clip, a_max=None)
        data.append(values)

    # Create a DataFrame
    df = pd.DataFrame(np.array(data).T, columns=['CUST_NUM'] + list(config.keys()))
    df = df.astype(int)

    return df


def average_noisy_configs(configs, std_multiplier):
    # Get the list of all keys
    all_keys = set().union(*[config.keys() for config in configs])

    # Initialize the result dictionary
    result_config = {}

    for key in all_keys:
        # Initialize variables to calculate the mean
        total_mean = 0
        total_std = 0
        total_clip = 0

        # Iterate over each config and accumulate the values for the key
        for config in configs:
            if key in config:
                total_mean += config[key].get('mean', 0)
                total_std += config[key].get('std', 0)
                total_clip += config[key].get('clip', 0)

        # Calculate the average values
        avg_mean = total_mean / len(configs)
        avg_std = total_std / len(configs) * std_multiplier
        avg_clip = total_clip / len(configs)

        # Create the average configuration for the key
        avg_config = {'mean': avg_mean, 'std': avg_std, 'clip': avg_clip}

        # Add the key and its average configuration to the result dictionary
        result_config[key] = avg_config

    return result_config


def generate_targets(df, target_config, quantile_thr, noise_scale):

    # Normalize the columns
    df_targets = df.copy()
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_targets), columns=df_targets.columns)

    # Compute score
    score = sum(df_normalized[col] * weight for col, weight in target_config.items())

    # Normalize churn score between 0 and 1
    min_max_scaler = MinMaxScaler()
    score_normalized = min_max_scaler.fit_transform(score.values.reshape(-1, 1))
    score_normalized = pd.Series(score_normalized.flatten(), index=score.index)

    # Add 10% noise to churn score
    noise = np.random.normal(scale=noise_scale, size=score.shape[0])
    score_with_noise = score_normalized + noise

    # Normalize churn score with noise between 0 and 1
    score_normalized_with_noise = min_max_scaler.fit_transform(score_with_noise.values.reshape(-1, 1))

    # Add churn score with noise as a new column
    df_targets['score'] = score_normalized_with_noise

    # Set the top X% of values to churn_target = 1
    threshold = df_targets['score'].quantile(quantile_thr)  # Adjust the quantile as needed
    df_targets['target'] = (df_targets['score'] >= threshold).astype(int)

    return df_targets['target'].values


def generate_request_dates(base_date):
    days_to_add = int(np.random.normal(loc=25, scale=5))  # Adjust loc and scale as needed
    return base_date + timedelta(days=days_to_add)


def generate_dates_between(start_date_str, end_date_str):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    return date_range.strftime('%Y-%m-%d').tolist()


def format_requests(df, request_columns):

    df_requests = df[request_columns]

    # assign a request date for the target
    for target in request_columns[2:]:
        df_requests.loc[df_requests[target] == 1, f'{target}_date'] = df_requests.loc[df_requests[target] == 1, 'date'].apply(generate_request_dates)

    # Convert the generated dates to datetime64[ns] format
    date_columns = [f'{target}_date' for target in request_columns[2:]]
    df_requests[date_columns] = df_requests[date_columns].astype('datetime64[ns]')

    # Select relevant columns
    df_requests = df_requests[['CUST_NUM'] + date_columns]

    # Melt dataframe
    df_requests = pd.melt(df_requests, id_vars=['CUST_NUM'], var_name='EVENT_TYPE', value_name='DATE')
    df_requests = df_requests.dropna()
    
    # clean event_type column values
    df_requests['EVENT_TYPE'] = np.where(df_requests['EVENT_TYPE'].str.contains('churn'), 'churn_request', df_requests['EVENT_TYPE'])
    df_requests['EVENT_TYPE'] = df_requests['EVENT_TYPE'].str.replace(r'(.*)_target_date', r'\1 _package_purchase')
    df_requests = df_requests.sort_values('DATE')

    # Remove requests that are too recent - 15 days lag period in the systems
    df_requests = df_requests[df_requests['DATE'] < (datetime.now() - timedelta(days=15))]

    return df_requests


def generate_datasets(start_date, end_date, folder_name, population_distribution_config, target_definition_config, storage_config, global_config):

    main_path = storage_config['main_path']
    container_name = storage_config['container_name']
    storage_account_key = storage_config['storage_account_key']
    storage_account_name = storage_config['storage_account_name']

    blob_path = f'{container_name}/{main_path}/{folder_name}'


    dates = generate_dates_between(start_date, end_date)
    df_list = []
    for date in dates:
        df_churn_list = []
        for pop_name, pop_config in population_distribution_config['churn'].items():
            df_churn_list.append(generate_data(pop_config, num_rows=global_config['daily_nb_rows_per_group']))
        # add noise
        df_churn_list.append(generate_noisy_data(configs=population_distribution_config['churn'], std_multiplier=1, 
                                                num_rows_per_dataset=int(global_config['daily_nb_rows_per_group']*global_config['noise_per_population'])))

        df_sport_list = []
        for pop_name, pop_config in population_distribution_config['sport'].items():
            df_sport_list.append(generate_data(pop_config, num_rows=global_config['daily_nb_rows_per_group']))
        # add noise
        df_sport_list.append(generate_noisy_data(configs=population_distribution_config['sport'], std_multiplier=1, 
                                                num_rows_per_dataset=int(global_config['daily_nb_rows_per_group']*global_config['noise_per_population'])))

        df_cinema_list = []
        for pop_name, pop_config in population_distribution_config['cinema'].items():
            df_cinema_list.append(generate_data(pop_config, num_rows=global_config['daily_nb_rows_per_group']))
        # add noise
        df_cinema_list.append(generate_noisy_data(configs=population_distribution_config['cinema'], std_multiplier=1, 
                                                num_rows_per_dataset=int(global_config['daily_nb_rows_per_group']*global_config['noise_per_population'])))

        df_churn = pd.concat(df_churn_list, axis=0)
        df_sport = pd.concat(df_sport_list, axis=0)
        df_cinema = pd.concat(df_cinema_list, axis=0)

        # perform a shuffle to mix the sub-distributions and add to list
        df_day = pd.concat([df_churn, df_sport, df_cinema], axis=0).sample(frac=1)

        df_day['churn_target'] = generate_targets(df=df_day, target_config=target_definition_config['churn'], quantile_thr=(1-global_config['ratio_churn']), noise_scale=global_config['target_noise_scale'])
        df_day['sport_target'] = generate_targets(df=df_day, target_config=target_definition_config['sport_bundle'], quantile_thr=(1-global_config['ratio_sport']), noise_scale=global_config['target_noise_scale'])
        df_day['cinema_target'] = generate_targets(df=df_day, target_config=target_definition_config['cinema_bundle'], quantile_thr=(1-global_config['ratio_cinema']), noise_scale=global_config['target_noise_scale'])

        df_day['date'] = date
        df_list.append(df_day)

    df = pd.concat(df_list, axis=0)
    df['date'] = pd.to_datetime(df['date'])

    # save socio demo data
    df_socio_demo = df[global_config['groups']['socio_demo']['cols']]
    save_to_blob(df_socio_demo, blob_path, 'socio_demo.parquet', storage_account_key, storage_account_name)

    # save internet data
    df_internet = df[global_config['groups']['internet']['cols']]
    save_to_blob(df_internet, blob_path, 'internet.parquet', storage_account_key, storage_account_name)

    # save TV data
    df_tv = df[global_config['groups']['tv']['cols']]
    save_to_blob(df_tv, blob_path, 'tv.parquet', storage_account_key, storage_account_name)

    # save mobile data - to have mobile data for a customer, the customer must have NB_MOBILE_SUBS > 0
    df_mobile = df[global_config['groups']['mobile']['cols']]
    df_mobile[df_mobile['NB_MOBILE_SUBS'] > 0]
    save_to_blob(df_mobile, blob_path, 'mobile.parquet', storage_account_key, storage_account_name)

    # save journeys data - to have jounrey data about a customer, the customer must have at leats one commplain
    df_journeys = df[global_config['groups']['journeys']['cols']]
    df_journeys = df_journeys[df_journeys[global_config['groups']['journeys']['cols'][2:]].sum(axis=1) > 0]
    save_to_blob(df_journeys, blob_path, 'journeys.parquet', storage_account_key, storage_account_name)

    # save requests data
    df_requests = format_requests(df, global_config['groups']['requests']['cols'])
    save_to_blob(df_requests, blob_path, 'requests.parquet', storage_account_key, storage_account_name)

    Logger.info(f"Data saved from {start_date} to {end_date}")