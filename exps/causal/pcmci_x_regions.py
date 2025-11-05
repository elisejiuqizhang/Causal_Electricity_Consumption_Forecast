# testing inter-region causality on meteo variables
# allowing a list of city/regions names, and one variable

import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import contextlib


from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars
from utils.causality.pcmci import create_pcmci

# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tigramite import plotting as tp

# Paths
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'PCMCI_x_regions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(ROOT, 'data', 'ieso_era5')
DATA_FILE_PREFIX='combined_ieso_era5_avg_'

SCALING='standard'  # 'minmax' or 'standard' or None

# Choose list of regions/cities, and one variable of interest
NAME_VAR='t2m_degC' # available options: "t2m_degC", "d2m_degC", "tp_mm", "tcw", "tcc", "skt", "avg_snlwrf", "avg_snswrf"
list_regions=dict_regions.keys()# available options: keys of dict_regions, or cities in list_cities

OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'{NAME_VAR}', '_'.join(list_regions))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PCMCI parameters
list_time_lags = range(6, 7)  # hourly lags (the tau_max for pcmci)
pc_alpha=0.05
list_plot_thres = np.arange(0.025, 0.5, 0.025)

# load data
list_dfs = []
# load the target variable from each region/city, create a new df with columns named after the city
for region in list_regions:
    print(f"Loading {NAME_VAR} for region:{region}")
    if region in dict_regions:
        if len(dict_regions[region])>1: # need to do each city and then average (since it's meteo variables, cannot aggregate)
            list_tmp_region_dfs = []
            for city in dict_regions[region]:
                print(f'  Loading city: {city}')
                data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{city.replace(" ", "_").lower()}.csv')
                city_df = pd.read_csv(data_file, parse_dates=['time'])
                city_df = city_df[['time', NAME_VAR]]
                city_df.set_index('time', inplace=True)
                city_df = city_df.rename(columns={NAME_VAR: f'{region}_{city}'})
                list_tmp_region_dfs.append(city_df)
            # average over cities to get the region-level meteo reading
            df_region = pd.concat(list_tmp_region_dfs, axis=1)
            df_region = df_region.mean(axis=1).to_frame()
            # rename column
            df_region = df_region.rename(columns={0: f'{region}'})

        else: # only one city in the region
            data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{dict_regions[region][0].replace(" ", "_").lower()}.csv')
            df_region = pd.read_csv(data_file, parse_dates=['time'])
            df_region = df_region[['time', NAME_VAR]]
            df_region.set_index('time', inplace=True)
            df_region = df_region.rename(columns={NAME_VAR: f'{region}'})
        
        list_dfs.append(df_region)

    else: 
        if region in list_cities: # single city
            print(f'  Loading city: {region}')
            data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{region.replace(" ", "_").lower()}.csv')
            city_df = pd.read_csv(data_file, parse_dates=['time'])
            city_df = city_df[['time', NAME_VAR]]
            city_df.set_index('time', inplace=True)
            city_df = city_df.rename(columns={NAME_VAR: f'{region}'})
            list_dfs.append(city_df)
        else:
            raise ValueError(f'Unknown region/city name: {region}')
        
# merge all dfs on time
df_merged = list_dfs[0]
for df in list_dfs[1:]:
    df_merged = pd.merge(df_merged, df, on='time', how='inner')


# scale by column if needed
if SCALING is not None:
    for var in df_merged.columns:
        if SCALING.lower()=='minmax':
            scaler = MinMaxScaler()
        elif SCALING.lower()=='standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f'Unknown scaling method: {SCALING}')
        df_merged[var] = scaler.fit_transform(df_merged[[var]])

# PCMCI for the merged df
for lag in list_time_lags:
    pcmci=create_pcmci(df_merged, var_names=df_merged.columns.tolist(), robust=True, wls=False)
    with open(os.path.join(OUTPUT_DIR, f"pcmci_maxLag{lag}_results.txt"), "w") as f:
        with contextlib.redirect_stdout(f):
            results=pcmci.run_pcmci(pc_alpha=pc_alpha, tau_max=lag)
    # save the results
    with open(os.path.join(OUTPUT_DIR, f"pcmci_maxLag{lag}_pValues.npy"), "wb") as f:
        f.write(results['p_matrix'].round(3))
    with open(os.path.join(OUTPUT_DIR, f"pcmci_maxLag{lag}_mci_parCorr.npy"), "wb") as f:
        f.write(results['val_matrix'].round(3))

    p=results['p_matrix']
    val=results['val_matrix']
    graph = results['graph'].copy()

    # plot the results with different value thresholds
    for plot_thres in list_plot_thres:
        # keep links that are both statistically significant and have a high enough MCI value
        keep = (p < pc_alpha) & (np.abs(val) >= plot_thres)

        # build a pruned graph
        graph_pruned = np.full_like(graph, '', dtype=object)
        graph_pruned[keep] = graph[keep]

        tp.plot_time_series_graph(
            figsize=(16, 12),
            graph=graph_pruned,
            val_matrix=val,
            var_names=df_merged.columns.tolist(),
            link_colorbar_label='MCI',
        )
        plt.savefig(os.path.join(OUTPUT_DIR, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_time_series_graph.png'))
        plt.close()

        tp.plot_graph(
            figsize=(16, 12),
            graph=graph_pruned,
            val_matrix=val,
            var_names=df_merged.columns.tolist(),
            link_colorbar_label='MCI',
        )
        plt.savefig(os.path.join(OUTPUT_DIR, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_graph.png'))
        plt.close()