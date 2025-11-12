"""
GRU Model OOD Inference Script
Load pretrained GRU models and test them on identified OOD weather windows
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from utils.forecast.rnn import GRUForecaster
from utils.forecast.construct_seq import construct_sequences_uni_output
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_cities import list_cities, dict_regions, list_vars
from utils.data_utils.info_features import list_F0, list_F1, list_F2, list_F3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(region, data_dir, data_file_prefix):
    """Load and prepare data for a specific region"""
    if region in dict_regions:
        if len(dict_regions[region]) > 1:
            list_tmp_region_dfs = []
            for city in dict_regions[region]:
                df_city = pd.read_csv(
                    os.path.join(data_dir, f'{data_file_prefix}{city.lower()}.csv'),
                    parse_dates=['time']
                ).set_index('time')
                df_city = df_city.add_prefix(f'{region}_{city}_')
                list_tmp_region_dfs.append(df_city)
            
            df_region = pd.concat(list_tmp_region_dfs, axis=1)
            list_agg_dfs = []
            for var in list_vars:
                cols_var = [col for col in df_region.columns if col.endswith(f'_{var}')]
                if var in ['TOTAL_CONSUMPTION', 'PREMISE_COUNT']:
                    list_agg_dfs.append(df_region[cols_var].sum(axis=1).to_frame(name=var))
                else:
                    list_agg_dfs.append(df_region[cols_var].mean(axis=1).to_frame(name=var))
            df_region = pd.concat(list_agg_dfs, axis=1)
        else:
            city = dict_regions[region][0]
            df_region = pd.read_csv(
                os.path.join(data_dir, f'{data_file_prefix}{city.lower()}.csv'),
                parse_dates=['time']
            ).set_index('time')
    else:
        if region in list_cities:
            df_region = pd.read_csv(
                os.path.join(data_dir, f'{data_file_prefix}{region.lower()}.csv'),
                parse_dates=['time']
            ).set_index('time')
        else:
            raise ValueError(f'Unknown region: {region}')
    
    # Drop AVG_CONSUMPTION_PER_PREMISE and add datetime features
    df_region = df_region.drop(columns=['AVG_CONSUMPTION_PER_PREMISE'], errors='ignore')
    df_region = time_features(df_region.reset_index(), time_col='time', drop_original=False).set_index('time')
    
    return df_region


def load_ood_windows(ood_file):
    """Load OOD windows from CSV file"""
    ood_df = pd.read_csv(ood_file)
    ood_df['start_datetime'] = pd.to_datetime(ood_df['start_datetime'])
    ood_df['end_datetime'] = pd.to_datetime(ood_df['end_datetime'])
    return ood_df


def load_model_and_scaler(model_path, scaler_path, input_size, d_model, n_layers, horizon, dropout=0.1):
    """Load pretrained GRU model and scaler"""
    model = GRUForecaster(
        input_size=input_size,
        hidden_size=d_model,
        num_layers=n_layers,
        dropout=dropout,
        output_size=1,
        horizon=horizon
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load scaler if exists
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler


def predict_on_window(model, df_window, scaler, input_length, horizon, stride, batch_size=64):
    """Make predictions on a specific time window"""
    # Normalize if scaler exists
    df_normalized = df_window.copy().astype('float32')
    if scaler is not None:
        df_normalized[df_normalized.columns] = scaler.transform(df_normalized)
    
    # Construct sequences
    X_arr, Y_arr = construct_sequences_uni_output(
        df_normalized, 
        history_len=input_length, 
        horizon=horizon, 
        step_size=stride, 
        target_col='TOTAL_CONSUMPTION'
    )
    
    if len(X_arr) == 0:
        return None, None, None
    
    # Create dataloader
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(Y_arr, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Get predictions
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)  # Shape: (B, 1, H)
            pred = pred.squeeze(1)  # Remove target dimension: (B, H)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(batch_Y.numpy())
    
    all_preds = np.vstack(all_preds)
    all_tgts = np.vstack(all_tgts)
    
    # Aggregate overlapping predictions
    num_windows = all_preds.shape[0]
    n_total = len(df_window)
    t_starts = np.arange(0, n_total - input_length - horizon + 1, stride)
    
    if len(t_starts) != num_windows:
        print(f"Warning: Number of windows mismatch: {len(t_starts)} vs {num_windows}")
        t_starts = t_starts[:num_windows]
    
    ts_pred = defaultdict(list)
    for w, t0 in enumerate(t_starts):
        for h in range(horizon):
            ts_pred[t0 + input_length + h].append(all_preds[w, h])
    
    pred_indices = sorted(ts_pred.keys())
    agg_pred = np.array([np.mean(ts_pred[i]) for i in pred_indices])
    
    # Denormalize predictions
    if scaler is not None:
        target_idx = df_window.columns.get_loc('TOTAL_CONSUMPTION')
        agg_pred = agg_pred * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    
    # Align with true values
    aligned_dt = df_window.index[pred_indices]
    true_aligned = df_window['TOTAL_CONSUMPTION'].values[pred_indices]
    
    return agg_pred, true_aligned, aligned_dt


def calculate_metrics(pred, true):
    """Calculate forecasting metrics"""
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    
    # MAPE
    mask = true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mape = np.nan
    
    # SMAPE
    denom = (np.abs(true) + np.abs(pred)) / 2
    mask_smape = denom != 0
    if mask_smape.sum() > 0:
        smape = np.mean(np.abs(true[mask_smape] - pred[mask_smape]) / denom[mask_smape]) * 100
    else:
        smape = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }


def main():
    parser = argparse.ArgumentParser(description='GRU OOD Inference')
    
    # Region and data
    parser.add_argument('--region', type=str, required=True, help='Region name')
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(ROOT_DIR, 'data', 'ieso_era5'),
                       help='Data directory')
    parser.add_argument('--data_file_prefix', type=str, 
                       default='combined_ieso_era5_avg_',
                       help='Data file prefix')
    
    # Model configuration
    parser.add_argument('--feature_set', type=str, default='F2', 
                       choices=['F0', 'F1', 'F2', 'F3'],
                       help='Feature set used during training')
    parser.add_argument('--model_dir', type=str, 
                       default=os.path.join(ROOT_DIR, 'outputs', 'forecast1111', 'per_region', 'gru_single_train'),
                       help='Model directory')
    parser.add_argument('--training_config', type=str,
                       default='bs64_ep500_lr0.0001_tr0.93_vr0.07_pat20_esep0.0001',
                       help='Training configuration folder name')
    parser.add_argument('--seed', type=int, default=97, help='Random seed used during training')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to use')
    
    # Model hyperparameters (must match training)
    parser.add_argument('--d_model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--input_length', type=int, default=168, help='Input sequence length')
    parser.add_argument('--horizon', type=int, default=24, help='Prediction horizon')
    parser.add_argument('--stride', type=int, default=1, help='Stride for predictions')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    
    # OOD windows
    parser.add_argument('--ood_file', type=str, required=True,
                       help='Path to OOD windows CSV file')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(ROOT_DIR, 'outputs', 'ood_inference', 'gru'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting OOD inference for {args.region}")
    print(f"Feature set: {args.feature_set}, Fold: {args.fold}")
    
    # Load data
    print("Loading data...")
    df_region = load_data(args.region, args.data_dir, args.data_file_prefix)
    
    # Select features
    if args.feature_set == 'F0':
        selected_features = list_F0
    elif args.feature_set == 'F1':
        selected_features = list_F1
    elif args.feature_set == 'F2':
        selected_features = list_F2
    elif args.feature_set == 'F3':
        selected_features = list_F3
    df_region = df_region[selected_features]
    
    # Load OOD windows
    print(f"Loading OOD windows from {args.ood_file}")
    ood_windows = load_ood_windows(args.ood_file)
    print(f"Found {len(ood_windows)} OOD windows")
    
    # Load model and scaler
    model_path = os.path.join(
        args.model_dir,
        args.region.replace(" ", "_"),
        args.training_config,
        args.feature_set,
        str(args.seed),
        f'fold_{args.fold}',
        'best_model.pth'
    )
    scaler_path = os.path.join(
        args.model_dir,
        args.region.replace(" ", "_"),
        args.training_config,
        args.feature_set,
        str(args.seed),
        f'fold_{args.fold}',
        'scaler_all.pkl'
    )
    
    print(f"Loading model from {model_path}")
    model, scaler = load_model_and_scaler(
        model_path, scaler_path,
        input_size=len(selected_features),
        d_model=args.d_model,
        n_layers=args.n_layers,
        horizon=args.horizon,
        dropout=args.dropout
    )
    
    # Process each OOD window
    results = []
    all_predictions = []
    
    for idx, window in ood_windows.iterrows():
        print(f"\nProcessing window {idx + 1}/{len(ood_windows)}: {window['start_datetime']} to {window['end_datetime']}")
        
        # Extract window data (need extra for input_length context)
        window_start = window['start_datetime'] - pd.Timedelta(hours=args.input_length)
        window_end = window['end_datetime']
        
        # Find the data in the dataframe
        mask = (df_region.index >= window_start) & (df_region.index <= window_end)
        df_window = df_region[mask].copy()
        
        if len(df_window) < args.input_length + args.horizon:
            print(f"  Warning: Not enough data for window (need {args.input_length + args.horizon}, got {len(df_window)})")
            continue
        
        # Make predictions
        pred, true, aligned_dt = predict_on_window(
            model, df_window, scaler,
            args.input_length, args.horizon, args.stride,
            args.batch_size
        )
        
        if pred is None:
            print(f"  Warning: Could not generate predictions for this window")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(pred, true)
        print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%, SMAPE: {metrics['SMAPE']:.2f}%")
        
        # Store results
        results.append({
            'window_idx': idx,
            'start_datetime': window['start_datetime'],
            'end_datetime': window['end_datetime'],
            'ood_fraction': window['ood_fraction'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'SMAPE': metrics['SMAPE'],
            'n_predictions': len(pred)
        })
        
        # Store detailed predictions
        for i, (dt, p, t) in enumerate(zip(aligned_dt, pred, true)):
            all_predictions.append({
                'window_idx': idx,
                'datetime': dt,
                'predicted_load': p,
                'true_load': t,
                'error': p - t,
                'abs_error': abs(p - t)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(all_predictions)
    
    output_prefix = f"{args.region}_{args.feature_set}_fold{args.fold}"
    results_file = os.path.join(args.output_dir, f'{output_prefix}_ood_metrics.csv')
    predictions_file = os.path.join(args.output_dir, f'{output_prefix}_ood_predictions.csv')
    
    results_df.to_csv(results_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print('='*60)
    print(f"Total OOD windows processed: {len(results_df)}")
    print(f"\nAverage metrics across all OOD windows:")
    print(f"  MAE:   {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}")
    print(f"  RMSE:  {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}")
    print(f"  MAPE:  {results_df['MAPE'].mean():.2f}% ± {results_df['MAPE'].std():.2f}%")
    print(f"  SMAPE: {results_df['SMAPE'].mean():.2f}% ± {results_df['SMAPE'].std():.2f}%")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f'{output_prefix}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"OOD Inference Results for {args.region}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: GRU\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"OOD windows file: {args.ood_file}\n")
        f.write(f"\nTotal OOD windows processed: {len(results_df)}\n")
        f.write(f"\nAverage metrics across all OOD windows:\n")
        f.write(f"  MAE:   {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}\n")
        f.write(f"  RMSE:  {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}\n")
        f.write(f"  MAPE:  {results_df['MAPE'].mean():.2f}% ± {results_df['MAPE'].std():.2f}%\n")
        f.write(f"  SMAPE: {results_df['SMAPE'].mean():.2f}% ± {results_df['SMAPE'].std():.2f}%\n")
        f.write(f"\nPer-window results:\n")
        f.write(results_df.to_string())
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {predictions_file}")
    print(f"  - {summary_file}")


if __name__ == '__main__':
    main()
