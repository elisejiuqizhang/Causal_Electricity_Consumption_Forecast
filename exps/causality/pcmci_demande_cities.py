# pcmci, load all cities' electricity demand data (total, not the average one)
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import contextlib

from utils.data_utils.datetime_utils import add_hourly_calendar_features
from utils.data_utils.info_cities import list_cities, dict_city_coords
from utils.data_utils.processing import min_max_scale, corr
from utils.causality.pcmci import create_pcmci

from tigramite import plotting as tp

#output dir
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_pcmci_demande_cities')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
IESO_DIR = os.path.join(ROOT, 'data', 'ieso_hourly')
list_dfs = []
for city_name in list_cities:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])
    ieso_df = ieso_df[['TIMESTAMP', 'TOTAL_CONSUMPTION']]
    ieso_df = ieso_df.rename(columns={'TOTAL_CONSUMPTION': city_name})
    list_dfs.append(ieso_df)
df = list_dfs[0]
for other_df in list_dfs[1:]:
    df = pd.merge(df, other_df, on='TIMESTAMP', how='inner')

df = df.set_index('TIMESTAMP')

# print(df.head())
# print(list(df.columns))


# pcmci
time_lag_max = 3  # hours
pc_alpha = 0.05

var_names = list(df.columns)
pcmci = create_pcmci(df, var_names=var_names, time_lag_max=time_lag_max, robust=True, wls=False)

# file save name
file_save_name = f'{city_name.replace(" ", "_")}_lag{time_lag_max}'
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



# link_output_file = os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{pc_alpha}_link_output.txt')
# with open(link_output_file, 'w') as f:
#     with contextlib.redirect_stdout(f):
#         pcmci.print_significant_links(p_matrix = results['p_matrix'],val_matrix = results['val_matrix'],
#                                         alpha_level = pc_alpha)
    
#     # plot the graph based on results[0], which is an array of shape [N, N, tau_max+1]
# tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names)
# plt.savefig(os.path.join(OUTPUT_DIR, file_save_name+f'_alpha{pc_alpha}_graph.png'))
# plt.close()

# tp.plot_time_series_graph(
#     figsize=(12, 9),
#     val_matrix=results['val_matrix'],
#     graph=results['graph'],
#     var_names=var_names,
#     link_colorbar_label='MCI',
# )
# plt.savefig(os.path.join(OUTPUT_DIR, file_save_name+f'_alpha{pc_alpha}_time_series_graph.png'))
# plt.close()




# alpha = pc_alpha
# min_abs_val = 0.05  # my cutoff value for cleaning up plots (eliminating weak links)

# p = results['p_matrix']
# val = results['val_matrix']
# graph = results['graph'].copy()

# # Keep links that are BOTH statistically significant and strong enough
# keep = (p <= alpha) & (np.abs(val) >= min_abs_val)

# # Build a pruned graph: empty string means "no link" for the plotters
# graph_pruned = np.full_like(graph, '', dtype=object)
# graph_pruned[keep] = graph[keep]

# # (Optional) if you want the time-series graph as well, reuse graph_pruned
# tp.plot_graph(graph=graph_pruned, val_matrix=val, var_names=var_names,
#               link_colorbar_label='MCI', figsize=(12, 9))
# plt.savefig(os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{alpha}_min{min_abs_val}_graph.png'))
# plt.close()

# tp.plot_time_series_graph(
#     val_matrix=val, graph=graph_pruned, var_names=var_names,
#     figsize=(12, 9), link_colorbar_label='MCI'
# )
# plt.savefig(os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{alpha}_min{min_abs_val}_tsgraph.png'))
# plt.close()


alpha= pc_alpha
list_min_abs_val = [0.05, 0.10, 0.15]  # my cutoff values for cleaning up plots (eliminating weak links)

p = results['p_matrix']
val = results['val_matrix']
graph = results['graph'].copy()

for min_abs_val in list_min_abs_val:
    # Keep links that are BOTH statistically significant and strong enough
    keep = (p <= alpha) & (np.abs(val) >= min_abs_val)

    # Build a pruned graph: empty string means "no link" for the plotters
    graph_pruned = np.full_like(graph, '', dtype=object)
    graph_pruned[keep] = graph[keep]

    # (Optional) if you want the time-series graph as well, reuse graph_pruned
    tp.plot_graph(graph=graph_pruned, val_matrix=val, var_names=var_names,
                  link_colorbar_label='MCI', figsize=(12, 9))
    plt.savefig(os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{alpha}_min{min_abs_val}_graph.png'))
    plt.close()

    tp.plot_time_series_graph(
        val_matrix=val, graph=graph_pruned, var_names=var_names,
        figsize=(12, 9), link_colorbar_label='MCI'
    )
    plt.savefig(os.path.join(OUTPUT_DIR, file_save_name + f'_alpha{alpha}_min{min_abs_val}_tsgraph.png'))
    plt.close()