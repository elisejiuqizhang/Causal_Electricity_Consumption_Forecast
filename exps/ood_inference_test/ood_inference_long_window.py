#!/usr/bin/env python3
"""
OOD Inference for Models Trained on Long Window
================================================

Evaluate models trained on 2018-01-01 to 2023-03-10 on OOD windows from test period.
Tests all feature sets (F0, F1, F2, F3) on the same trained model.

Usage:
    python ood_inference_long_window.py --model_type gru --region Toronto
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.forecast.set_seed import set_seed
from utils.forecast.construct_seq import construct_sequences_uni_output
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_ieso_vars
from utils.data_utils.info_features import list_F0, list_F1, list_F2, list_F3
from utils.forecast.rnn import GRUForecaster
from utils.forecast.TCN import TCNModel
from utils.forecast.PatchTST import PatchTSTModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments
parser = argparse.ArgumentParser(description='OOD Inference for Long Window Models')
parser.add_argument('--model_type', type=str, required=True, choices=['gru', 'tcn', 'patchtst'])
parser.add_argument('--region', type=str, required=True, choices=list(dict_regions.keys())+list_cities)
parser.add_argument('--seed', type=int, default=97)
parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data', 'ieso_era5'))
parser.add_argument('--data_file_prefix', type=str, default='combined_ieso_era5_avg_')
parser.add_argument('--test_start', type=str, default='2023-03-11')
parser.add_argument('--test_end', type=str, default='2024-03-10')
parser.add_argument('--input_length', type=int, default=168)
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--scaler', type=str, default='standard')

args = parser.parse_args()
set_seed(args.seed)

print("=" * 80)
print(f"OOD Inference - Long Window Models")
print("=" * 80)
print(f"Model: {args.model_type.upper()}")
print(f"Region: {args.region}")
print(f"Test Period: {args.test_start} to {args.test_end}")
print("=" * 80)

# Feature sets to test
FEATURE_SETS = ['F0', 'F1', 'F2', 'F3']
feature_map = {
    'F0': list_F0,
    'F1': list_F1,
    'F2': list_F2,
    'F3': list_F3
}

# Load OOD windows
ood_windows_file = os.path.join(ROOT_DIR, 'outputs', 'ood_analysis_test', f'ood_windows_{args.region}_test.csv')
ood_windows = pd.read_csv(ood_windows_file)
ood_windows['start_datetime'] = pd.to_datetime(ood_windows['start_datetime'])
ood_windows['end_datetime'] = pd.to_datetime(ood_windows['end_datetime'])
print(f"\n✅ Loaded {len(ood_windows)} OOD windows for {args.region}")

# Load test period data
print(f"\nLoading test data for {args.region}...")
if args.region in dict_regions:
    if len(dict_regions[args.region]) > 1:  # Multi-city region
        list_tmp_region_dfs = []
        for city in dict_regions[args.region]:
            data_file = os.path.join(args.data_dir, f'{args.data_file_prefix}{city.replace(" ", "_").lower()}.csv')
            city_df = pd.read_csv(data_file)
            time_col = 'time' if 'time' in city_df.columns else 'datetime_utc'
            city_df[time_col] = pd.to_datetime(city_df[time_col])
            city_df = city_df[[time_col] + list_vars]
            city_df.set_index(time_col, inplace=True)
            city_df = city_df.rename(columns={var: f'{args.region}_{city}_{var}' for var in list_vars})
            list_tmp_region_dfs.append(city_df)
        
        df_region = pd.concat(list_tmp_region_dfs, axis=1)
        list_agg_dfs = []
        for var in list_vars:
            cols_var = [col for col in df_region.columns if col.endswith(f'_{var}')]
            if var in list_ieso_vars:
                df_var = df_region[cols_var].sum(axis=1).to_frame(name=f'{args.region}_{var}')
            else:
                df_var = df_region[cols_var].mean(axis=1).to_frame(name=f'{args.region}_{var}')
            list_agg_dfs.append(df_var)
        df_region = pd.concat(list_agg_dfs, axis=1)
        df_region = df_region.rename(columns={col: '_'.join(col.split('_')[1:]) for col in df_region.columns})
    else:  # Single city region
        data_file = os.path.join(args.data_dir, f'{args.data_file_prefix}{dict_regions[args.region][0].replace(" ", "_").lower()}.csv')
        df_region = pd.read_csv(data_file)
        time_col = 'time' if 'time' in df_region.columns else 'datetime_utc'
        df_region[time_col] = pd.to_datetime(df_region[time_col])
        df_region = df_region[[time_col] + list_vars]
        df_region.set_index(time_col, inplace=True)
else:
    if args.region in list_cities:
        data_file = os.path.join(args.data_dir, f'{args.data_file_prefix}{args.region.replace(" ", "_").lower()}.csv')
        city_df = pd.read_csv(data_file)
        time_col = 'time' if 'time' in city_df.columns else 'datetime_utc'
        city_df[time_col] = pd.to_datetime(city_df[time_col])
        city_df = city_df[[time_col] + list_vars]
        city_df.set_index(time_col, inplace=True)
        df_region = city_df
    else:
        raise ValueError(f'Unknown region/city name: {args.region}')

# Drop AVG_CONSUMPTION_PER_PREMISE and add datetime features
df_region = df_region.drop(columns=['AVG_CONSUMPTION_PER_PREMISE'], errors='ignore')
df_region = time_features(df_region.reset_index(), time_col=df_region.index.name, drop_original=False).set_index(df_region.index.name)

# Filter to test period
df_test = df_region[args.test_start:args.test_end].copy()
print(f"✅ Loaded {len(df_test)} test records")
print(f"   Date range: {df_test.index.min()} to {df_test.index.max()}")

# Output directory
output_dir = os.path.join(ROOT_DIR, 'outputs', 'ood_inference_test', f'{args.model_type}_long_window')
os.makedirs(output_dir, exist_ok=True)

# Results storage
all_results = []

# Loop through each feature set
for feature_set in FEATURE_SETS:
    print("\n" + "=" * 80)
    print(f"Testing Feature Set: {feature_set}")
    print("=" * 80)
    
    # Model directory
    model_name = f"{args.model_type}_single_train"
    model_dir = os.path.join(
        ROOT_DIR, 'outputs', 'forecast_long_window', 'per_region', model_name,
        args.region.replace(" ", "_"),
        'bs64_ep500_lr0.0001_longwindow',
        feature_set,
        str(args.seed)
    )
    
    # Check if model exists
    model_path = os.path.join(model_dir, 'best_model.pt')
    scaler_path = os.path.join(model_dir, 'scaler_all.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        continue
    
    print(f"✅ Loading model from: {model_dir}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler_all = pickle.load(f)
    
    # Select features for this feature set
    selected_features = feature_map[feature_set]
    df_test_fs = df_test[selected_features].copy()
    
    # Scale data
    df_test_scaled = df_test_fs.copy()
    df_test_scaled = df_test_scaled.astype('float32')
    df_test_scaled.loc[:, :] = scaler_all.transform(df_test_scaled.values)
    
    # Build sequences for entire test period
    X_test, Y_test = construct_sequences_uni_output(
        df_test_scaled,
        history_len=args.input_length,
        horizon=args.horizon,
        step_size=1,
        target_col='TOTAL_CONSUMPTION'
    )
    
    # Create index mapping from sequence to original timestamps
    seq_start_indices = np.arange(len(df_test_scaled) - args.input_length - args.horizon + 1)
    seq_end_indices = seq_start_indices + args.input_length + args.horizon - 1
    seq_timestamps = df_test_scaled.index[seq_start_indices + args.input_length - 1]
    
    print(f"  Test sequences: {X_test.shape}")
    
    # Load model
    n_features = X_test.shape[1]
    
    if args.model_type == 'gru':
        model = GRUForecaster(
            input_size=n_features,
            hidden_size=64,
            num_layers=4,
            output_size=1,
            horizon=args.horizon,
            dropout=0.1
        ).to(device)
    elif args.model_type == 'tcn':
        model = TCNModel(
            input_channels=X_test.shape[1],
            output_horizon=args.horizon,
            num_targets=1,
            hidden_channels=64,
            levels=4,
            kernel_size=3,
            dilation_base=2,
            dropout=0.1
        ).to(device)
    elif args.model_type == 'patchtst':
        model = PatchTSTModel(
            input_channels=X_test.shape[1],
            output_horizon=args.horizon,
            num_targets=1,
            context_length=args.input_length,
            d_model=64,
            n_heads=4,
            n_layers=3,
            patch_len=16,
            patch_stride=8,
            d_ff=256,
            dropout=0.1
        ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded")
    
    # Get predictions for all test sequences
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_pred = model(X_batch)
            
            if Y_pred.dim() == 3 and Y_pred.size(1) == 1:
                Y_pred = Y_pred.squeeze(1)
            
            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(Y_batch.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Denormalize predictions and targets
    # Get the scaler mean and std for TOTAL_CONSUMPTION
    consumption_idx = selected_features.index('TOTAL_CONSUMPTION')
    if isinstance(scaler_all, StandardScaler):
        mean = scaler_all.mean_[consumption_idx]
        scale = scaler_all.scale_[consumption_idx]
    elif isinstance(scaler_all, MinMaxScaler):
        min_val = scaler_all.data_min_[consumption_idx]
        range_val = scaler_all.data_range_[consumption_idx]
        mean = min_val
        scale = range_val
    
    if isinstance(scaler_all, StandardScaler):
        all_preds_denorm = all_preds * scale + mean
        all_targets_denorm = all_targets * scale + mean
    elif isinstance(scaler_all, MinMaxScaler):
        all_preds_denorm = all_preds * scale + mean
        all_targets_denorm = all_targets * scale + mean
    
    # Evaluate on each OOD window
    print(f"\nEvaluating on {len(ood_windows)} OOD windows...")
    
    for idx, window in ood_windows.iterrows():
        start_dt = window['start_datetime']
        end_dt = window['end_datetime']
        
        # Find sequences that fall within this OOD window
        # A sequence's prediction window should overlap with the OOD window
        mask = (seq_timestamps >= start_dt - pd.Timedelta(hours=args.horizon)) & \
               (seq_timestamps <= end_dt)
        
        if mask.sum() == 0:
            continue
        
        window_preds = all_preds_denorm[mask]
        window_targets = all_targets_denorm[mask]
        
        # Calculate metrics
        mae = np.mean(np.abs(window_preds - window_targets))
        rmse = np.sqrt(np.mean((window_preds - window_targets) ** 2))
        mape = np.mean(np.abs((window_targets - window_preds) / (window_targets + 1e-8))) * 100
        smape = np.mean(2 * np.abs(window_preds - window_targets) / (np.abs(window_preds) + np.abs(window_targets) + 1e-8)) * 100
        
        all_results.append({
            'model': args.model_type,
            'region': args.region,
            'feature_set': feature_set,
            'window_idx': idx,
            'start_datetime': start_dt,
            'end_datetime': end_dt,
            'ood_fraction': window['ood_fraction'],
            'n_sequences': mask.sum(),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape
        })
    
    print(f"✅ Evaluated {len([r for r in all_results if r['feature_set'] == feature_set])} windows for {feature_set}")

# Save results
results_df = pd.DataFrame(all_results)
output_file = os.path.join(output_dir, f'ood_results_{args.region}.csv')
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("✅ OOD Inference Complete!")
print("=" * 80)
print(f"Results saved to: {output_file}")
print("\nSummary by Feature Set:")
print("-" * 80)
for fs in FEATURE_SETS:
    fs_results = results_df[results_df['feature_set'] == fs]
    if len(fs_results) > 0:
        print(f"{fs}: MAE={fs_results['mae'].mean():.2f}, "
              f"RMSE={fs_results['rmse'].mean():.2f}, "
              f"MAPE={fs_results['mape'].mean():.2f}%, "
              f"SMAPE={fs_results['smape'].mean():.2f}%")
print("=" * 80)
