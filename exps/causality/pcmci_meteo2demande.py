import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import contextlib

from utils.data_utils.datetime_utils import add_hourly_calendar_features
from utils.data_utils.info_cities import list_cities, dict_city_coords
from utils.data_utils.processing import min_max_scale
from utils.causality.pcmci import create_pcmci

from tigramite import plotting as tp

#output dir
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_pcmci_meteo2demande')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# city of interest
city_idx=0 # between 0 and 24
city_name=list_cities[city_idx]
city_coords=dict_city_coords[city_name]

is_avg_ieso=True

# load data
ERA5_DIR = os.path.join(ROOT, 'data', 'era5')
IESO_DIR = os.path.join(ROOT, 'data', 'ieso_hourly')
era5_df = pd.read_csv(os.path.join(ERA5_DIR, f'{city_name.lower()}_era5_timeseries.csv.gz'), parse_dates=['time'])
if is_avg_ieso:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_avg_per_premise_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])
else:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])

# print(list(era5_df.columns)) #'time', 'tcw', 'tcc', 't2m_degC', 'u10_ms', 'v10_ms', 'd2m_degC', 'ssr', 'net_sw_Wm2', 'tp_mm', 'e_mm', 'skt', 'lw_down_Wm2', 'lw_up_Wm2', 'net_lw_Wm2', 'net_radiation_Wm2', 'city'
# print(list(ieso_df.columns)) #'TIMESTAMP', 'TOTAL_CONSUMPTION', 'PREMISE_COUNT'   or    'TIMESTAMP', 'TOTAL_CONSUMPTION', 'PREMISE_COUNT', 'AVG_CONSUMPTION_PER_PREMISE'

# variables of interest
meteo_vars = [
    'tp_mm',    
    'e_mm',         
    'tcw', 
    't2m_degC',# in m/s
    'u10_ms',# in m/s
    'v10_ms',# in m/s
    'net_radiation_Wm2'  # in W/m2
]

if is_avg_ieso:
    ieso_var = 'AVG_CONSUMPTION_PER_PREMISE'  # in kWh
else:
    ieso_var = 'TOTAL_CONSUMPTION'  # in kWh


# reduce to time+vars of interest
era5_df = era5_df[['time'] + meteo_vars]
ieso_df = ieso_df[['TIMESTAMP', ieso_var]]

# merge on time
df = pd.merge(era5_df, ieso_df, left_on='time', right_on='TIMESTAMP', how='inner')
df = df.drop(columns=['TIMESTAMP'])

# min_max scale all variables
for col in df.columns:
    if col != 'time':
        df[col], min_val, max_val = min_max_scale(df[col])

# # examine for NaNs
# print(f'Number of NaNs in the merged dataframe for {city_name}:')
# print(df.isna().sum())
# # locate NaNs
# nan_indices = df[df.isna().any(axis=1)].index
# if len(nan_indices) > 0:
#     print(f'Found {len(nan_indices)} rows with NaNs in the merged dataframe for {city_name}. Dropping these rows.')
#     df = df.drop(index=nan_indices)

# choose index between 1 and 51738 (inclusive) - from reading the nan_indices list
df = df[1:51739]  # to avoid any potential issues with lag 0 in PCMCI


# pcmci
time_lag_max = 1  # hours
pc_alpha = 0.05

var_names = meteo_vars + [ieso_var]
pcmci = create_pcmci(df.drop(columns=['time']), var_names=var_names, time_lag_max=time_lag_max, robust=True, wls=False)

# file save name
file_save_name = f'{city_name.replace(" ", "_")}_lag{time_lag_max}_avg_{is_avg_ieso}'
output_file = os.path.join(OUTPUT_DIR, file_save_name + '_output.txt')
with open(output_file, 'w') as f:
    with contextlib.redirect_stdout(f):
        if time_lag_max>1:
            results = pcmci.run_lpcmci(tau_max=time_lag_max, pc_alpha=pc_alpha)
        else:
            results = pcmci.run_pcmci(tau_max=time_lag_max, pc_alpha=pc_alpha)
# save the results (p_values, mci_parCorr)
with open(os.path.join(OUTPUT_DIR, file_save_name+'_p-values.npy'), 'wb') as f:
    f.write(results['p_matrix'].round(3))
with open(os.path.join(OUTPUT_DIR, file_save_name+'_mci_parCorr.npy'), 'wb') as f:
    f.write(results['val_matrix'].round(3))

link_output_file = os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{pc_alpha}_link_output.txt')
with open(link_output_file, 'w') as f:
    with contextlib.redirect_stdout(f):
        pcmci.print_significant_links(p_matrix = results['p_matrix'],val_matrix = results['val_matrix'],
                                        alpha_level = pc_alpha)
    
    # plot the graph based on results[0], which is an array of shape [N, N, tau_max+1]
tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names)
plt.savefig(os.path.join(OUTPUT_DIR, file_save_name+f'_alpha{pc_alpha}_graph.png'))
plt.close()

tp.plot_time_series_graph(
    figsize=(12, 9),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI',
)
plt.savefig(os.path.join(OUTPUT_DIR, file_save_name+f'_alpha{pc_alpha}_time_series_graph.png'))
plt.close()