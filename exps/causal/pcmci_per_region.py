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
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'PCMCI_per_region')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(ROOT, 'data', 'ieso_era5')
DATA_FILE_PREFIX='combined_ieso_era5_avg_'

SCALING='standard'  # 'minmax' or 'standard' or None

# PCMCI parameters
list_time_lags = range(1, 9)  # hourly lags (the tau_max for pcmci)
pc_alpha=0.05
list_plot_thres = np.arange(0.025, 0.5, 0.025)

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

            

            # process for the city
            for lag in list_time_lags:
                print(f'Processing lag: {lag}')
                pcmci=create_pcmci(city_df, var_names=list_vars, time_lag_max=lag, robust=True, wls=False)
                with open(os.path.join(save_dir_city, f"pcmci_maxLag{lag}_results.txt"), "w") as f:
                    with contextlib.redirect_stdout(f):
                        results=pcmci.run_pcmci(pc_alpha=pc_alpha)
                # save the results
                with open(os.path.join(save_dir_city, f"pcmci_maxLag{lag}_pValues.npy"), "wb") as f:
                    f.write(results['p_matrix'].round(3))
                with open(os.path.join(save_dir_city, f"pcmci_maxLag{lag}_mci_parCorr.npy"), "wb") as f:
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
                        var_names=list_vars,
                        link_colorbar_label='MCI',
                    )
                    plt.savefig(os.path.join(save_dir_city, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_time_series_graph.png'))
                    plt.close()

                    tp.plot_graph(
                        figsize=(16, 12),
                        graph=graph_pruned,
                        val_matrix=val,
                        var_names=list_vars,
                        link_colorbar_label='MCI',
                    )
                    plt.savefig(os.path.join(save_dir_city, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_graph.png'))
                    plt.close()

        # process for the aggregated region - for total electricity consumption, sum up; for other meteorological variables, take average
        df_region = pd.concat(list_dfs_region).groupby('time').agg({var: 'mean' if var in list_era5_vars else 'sum' for var in list_vars}).reset_index()
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

    for lag in list_time_lags:
        print(f'Processing lag: {lag}')
        pcmci=create_pcmci(df_region, var_names=list_vars, robust=True, wls=False)
        with open(os.path.join(save_dir_region, f"pcmci_maxLag{lag}_results.txt"), "w") as f:
            with contextlib.redirect_stdout(f):
                results=pcmci.run_pcmci(pc_alpha=pc_alpha, tau_max=lag)
        # save the results
        with open(os.path.join(save_dir_region, f"pcmci_maxLag{lag}_pValues.npy"), "wb") as f:
            f.write(results['p_matrix'].round(3))
        with open(os.path.join(save_dir_region, f"pcmci_maxLag{lag}_mci_parCorr.npy"), "wb") as f:
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
                var_names=list_vars,
                link_colorbar_label='MCI',
            )
            plt.savefig(os.path.join(save_dir_region, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_time_series_graph.png'))
            plt.close()

            tp.plot_graph(
                figsize=(16, 12),
                graph=graph_pruned,
                val_matrix=val,
                var_names=list_vars,
                link_colorbar_label='MCI',
            )
            plt.savefig(os.path.join(save_dir_region, f'pcmci_maxLag{lag}_alpha{pc_alpha}_thres{plot_thres:.3f}_graph.png'))
            plt.close()