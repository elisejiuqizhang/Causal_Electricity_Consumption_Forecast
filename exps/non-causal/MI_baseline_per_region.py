# mutual information regression based feature selection

import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import contextlib


from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Paths
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'non-causal-fs', 'MI_baseline_per_region')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(ROOT, 'data', 'ieso_era5')
DATA_FILE_PREFIX='combined_ieso_era5_avg_'

TGT_COL='TOTAL_CONSUMPTION'

SCALING='standard'  # 'minmax' or 'standard' or None

# threshold for feature selection
list_MI_thres = np.arange(0.0, 1, 0.025)

# load data
for region in dict_regions.keys():

    print(f'Processing region: {region}')
    save_dir_region = os.path.join(OUTPUT_DIR, region)
    os.makedirs(save_dir_region, exist_ok=True)

    if len(dict_regions[region])>1: # need to do each city and then also aggregate
        list_dfs_region = []
        for city in dict_regions[region]:
            print(f'Processing city: {city}')
            save_dir_city = os.path.join(save_dir_region, city)
            os.makedirs(save_dir_city, exist_ok=True)

            data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{city.replace(" ", "_").lower()}.csv')
            city_df = pd.read_csv(data_file, parse_dates=['time'])
            city_df = city_df[['time'] + list_vars]
            city_df.set_index('time', inplace=True)

            list_dfs_region.append(city_df.copy()) # get the original dataframe
            
            # scale by column if needed
            if SCALING is not None:
                for var in list_vars:
                    if SCALING=='minmax':
                        scaler = MinMaxScaler()
                    elif SCALING=='standard':
                        scaler = StandardScaler()
                    else:
                        raise ValueError(f'Unknown scaling method: {SCALING}')
                    city_df[var] = scaler.fit_transform(city_df[[var]])

            # compute MI for each variable with respect to target variable (electricity consumption)
            X = city_df.drop(columns=[TGT_COL])
            y = city_df[TGT_COL]
            mi_scores = mutual_info_regression(X, y)
            mi_dict = dict(zip(X.columns, mi_scores))

            # write the results to a txt with col name and MI score
            with open(os.path.join(save_dir_city, f'MI_scores.txt'), 'w') as f:
                f.write('Variable\tMI_Score\n')
                for var, score in mi_dict.items():
                    f.write(f'{var}\t{score}\n')

            # select features based on different thresholds and save to different files
            for thres in list_MI_thres:
                selected_vars = [var for var, score in mi_dict.items() if score >= thres]
                with open(os.path.join(save_dir_city, f'selected_features_MI_thres_{thres:.3f}.txt'), 'w') as f:
                    for var in selected_vars:
                        f.write(f'{var}\n')

        # process for the aggregated region - for total electricity consumption, sum up; for other meteorological variables, take average
        df_region = pd.concat(list_dfs_region).groupby('time').agg({var: 'mean' if var in list_era5_vars else 'sum' for var in list_vars}).reset_index()
        df_region.set_index('time', inplace=True)
    else:
        df_region = pd.read_csv(os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{region.replace(" ", "_").lower()}.csv'), parse_dates=['time'])
        df_region = df_region[['time'] + list_vars]
        df_region.set_index('time', inplace=True)

    # scale by column if needed
    if SCALING is not None:
        for var in list_vars:
            if SCALING=='minmax':
                scaler = MinMaxScaler()
            elif SCALING=='standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f'Unknown scaling method: {SCALING}')
            df_region[var] = scaler.fit_transform(df_region[[var]])

    # compute MI for each variable with respect to target variable (electricity consumption)    
    X = df_region.drop(columns=[TGT_COL])
    y = df_region[TGT_COL]
    mi_scores = mutual_info_regression(X, y)
    mi_dict = dict(zip(X.columns, mi_scores))
    # write the results to a txt with col name and MI score
    with open(os.path.join(save_dir_region, f'MI_scores.txt'), 'w') as f:
        f.write('Variable\tMI_Score\n')
        for var, score in mi_dict.items():
            f.write(f'{var}\t{score}\n')
    # select features based on different thresholds and save to different files
    for thres in list_MI_thres:
        selected_vars = [var for var, score in mi_dict.items() if score >= thres]
        with open(os.path.join(save_dir_region, f'selected_features_MI_thres_{thres:.3f}.txt'), 'w') as f:
            for var in selected_vars:
                f.write(f'{var}\n')

