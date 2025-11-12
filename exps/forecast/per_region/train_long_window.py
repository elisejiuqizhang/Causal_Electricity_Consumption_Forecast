#!/usr/bin/env python3
"""
Train models on long window (2018-01-01 to 2023-03-10) without validation split
================================================================================

This script trains forecasting models on the full training period without validation split.
The trained models will be used for OOD inference on the test period (2023-03-11 to 2024-03-10).

Usage:
    python train_long_window.py --model_type gru --region Toronto --feature_set F0
"""

import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'forecast_long_window', 'per_region')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import random, math, time
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.forecast.set_seed import set_seed
from utils.forecast.construct_seq import construct_sequences_uni_output
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars
from utils.data_utils.info_features import list_datetime, list_iseo_vars, list_F0, list_F1, list_F2, list_F3
from utils.forecast.rnn import GRUForecaster
from utils.forecast.TCN import TCNModel
from utils.forecast.PatchTST import PatchTSTModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments
import argparse
parser = argparse.ArgumentParser(description='Train models on long window (2018-01-01 to 2023-03-10)')

# Model and experiment settings
parser.add_argument('--model_type', type=str, required=True, choices=['gru', 'tcn', 'patchtst'], help='Model type')
parser.add_argument('--seed', type=int, default=97, help='Random seed')
parser.add_argument('--region', type=str, required=True, help='Region name', choices=list(dict_regions.keys())+list_cities)
parser.add_argument('--feature_set', type=str, required=True, help='Feature set', choices=['F0', 'F1', 'F2', 'F3'])

# Data settings
parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data', 'ieso_era5'), help='Data directory')
parser.add_argument('--data_file_prefix', type=str, default='combined_ieso_era5_avg_', help='Data file prefix')
parser.add_argument('--train_start', type=str, default='2018-01-01', help='Training start date')
parser.add_argument('--train_end', type=str, default='2023-03-10', help='Training end date')

# Scaling
parser.add_argument('--scaler', type=str, default='standard', choices=['minmax', 'standard', 'none'], help='Scaling method')

# Forecast parameters
parser.add_argument("--input_length", type=int, default=168, help="Input sequence length (168 = 1 week)")
parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon (24 hours)")
parser.add_argument("--stride", type=int, default=1, help="Stride between sequences")

# Training parameters (same as previous experiments)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
parser.add_argument("--early_stopping_eps", type=float, default=1e-4, help="Early stopping epsilon")

# Model hyperparameters (matching previous experiments)
# GRU
parser.add_argument("--gru_d_model", type=int, default=64, help="GRU hidden dimension")
parser.add_argument("--gru_n_layers", type=int, default=4, help="GRU number of layers")
parser.add_argument("--gru_dropout", type=float, default=0.1, help="GRU dropout")

# TCN
parser.add_argument("--tcn_hidden_channels", type=int, default=64, help="TCN hidden channels")
parser.add_argument("--tcn_levels", type=int, default=4, help="TCN levels")
parser.add_argument("--tcn_kernel_size", type=int, default=3, help="TCN kernel size")
parser.add_argument("--tcn_dropout", type=float, default=0.1, help="TCN dropout")

# PatchTST
parser.add_argument("--patchtst_d_model", type=int, default=64, help="PatchTST d_model")
parser.add_argument("--patchtst_n_heads", type=int, default=4, help="PatchTST n_heads")
parser.add_argument("--patchtst_n_layers", type=int, default=3, help="PatchTST n_layers")
parser.add_argument("--patchtst_patch_len", type=int, default=16, help="PatchTST patch_len")
parser.add_argument("--patchtst_stride", type=int, default=8, help="PatchTST stride")
parser.add_argument("--patchtst_dropout", type=float, default=0.1, help="PatchTST dropout")

args = parser.parse_args()

# Set seed
set_seed(args.seed)

# Setup output directory
model_name = f"{args.model_type}_single_train"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, model_name, f'{args.region.replace(" ", "_")}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create experiment name based on model type
if args.model_type == 'gru':
    exp_name = f"dm{args.gru_d_model}_nl{args.gru_n_layers}_inlen{args.input_length}_h{args.horizon}_scaler{args.scaler}"
elif args.model_type == 'tcn':
    exp_name = f"hc{args.tcn_hidden_channels}_lv{args.tcn_levels}_inlen{args.input_length}_h{args.horizon}_scaler{args.scaler}"
elif args.model_type == 'patchtst':
    exp_name = f"dm{args.patchtst_d_model}_nh{args.patchtst_n_heads}_nl{args.patchtst_n_layers}_pl{args.patchtst_patch_len}_ps{args.patchtst_stride}_inlen{args.input_length}_h{args.horizon}_scaler{args.scaler}"

training_info = f"bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_longwindow"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, training_info, args.feature_set, str(args.seed))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(f"Training {args.model_type.upper()} on Long Window")
print("=" * 80)
print(f"Region: {args.region}")
print(f"Feature Set: {args.feature_set}")
print(f"Training Period: {args.train_start} to {args.train_end}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)

# Tensorboard
TB_DIR = os.path.join(OUTPUT_DIR, 'tensorboard_runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter(log_dir=TB_DIR)

# Load data
print("Loading data...")
if args.region in dict_regions:
    if len(dict_regions[args.region]) > 1:  # Multi-city region
        list_tmp_region_dfs = []
        for city in dict_regions[args.region]:
            print(f'  Loading city: {city}')
            data_file = os.path.join(args.data_dir, f'{args.data_file_prefix}{city.replace(" ", "_").lower()}.csv')
            city_df = pd.read_csv(data_file)
            
            # Handle 'time' or 'datetime_utc' column
            time_col = 'time' if 'time' in city_df.columns else 'datetime_utc'
            city_df[time_col] = pd.to_datetime(city_df[time_col])
            city_df = city_df[[time_col] + list_vars]
            city_df.set_index(time_col, inplace=True)
            city_df = city_df.rename(columns={var: f'{args.region}_{city}_{var}' for var in list_vars})
            list_tmp_region_dfs.append(city_df)
        
        # Average over cities
        df_region = pd.concat(list_tmp_region_dfs, axis=1)
        list_agg_dfs = []
        for var in list_vars:
            cols_var = [col for col in df_region.columns if col.endswith(f'_{var}')]
            if var in list_ieso_vars:  # Sum for consumption and premise count
                df_var = df_region[cols_var].sum(axis=1).to_frame(name=f'{args.region}_{var}')
            else:  # Average for meteo variables
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

# Drop AVG_CONSUMPTION_PER_PREMISE
df_region = df_region.drop(columns=['AVG_CONSUMPTION_PER_PREMISE'], errors='ignore')

# Add datetime features
df_region = time_features(df_region.reset_index(), time_col=df_region.index.name, drop_original=False).set_index(df_region.index.name)

# Filter to training period
df_region = df_region[args.train_start:args.train_end]
print(f"✅ Loaded {len(df_region)} records")
print(f"   Date range: {df_region.index.min()} to {df_region.index.max()}")

# Select features
if args.feature_set == 'F0':
    selected_features = list_F0
elif args.feature_set == 'F1':
    selected_features = list_F1
elif args.feature_set == 'F2':
    selected_features = list_F2
elif args.feature_set == 'F3':
    selected_features = list_F3
else:
    raise ValueError(f'Unknown feature set: {args.feature_set}')

df_region = df_region[selected_features]
print(f"Selected features: {len(selected_features)}")

# Scale data
train_df = df_region.copy()
if args.scaler.lower() != 'none' and args.scaler is not None:
    if args.scaler.lower() == 'minmax':
        scaler_all = MinMaxScaler()
    elif args.scaler.lower() == 'standard':
        scaler_all = StandardScaler()
    else:
        raise ValueError(f'Unknown scaling method: {args.scaler}')
    
    train_df = train_df.astype('float32')
    scaler_all.fit(train_df.values)
    
    # Save scaler
    with open(os.path.join(OUTPUT_DIR, 'scaler_all.pkl'), 'wb') as f:
        pickle.dump(scaler_all, f)
    
    train_df.loc[:, :] = scaler_all.transform(train_df.values)
    print(f"✅ Applied {args.scaler} scaling")

# Construct sequences
print("Constructing sequences...")
X_arr, Y_arr = construct_sequences_uni_output(
    train_df, 
    history_len=args.input_length, 
    horizon=args.horizon, 
    step_size=args.stride, 
    target_col='TOTAL_CONSUMPTION'
)
print(f"  X shape: {X_arr.shape}, Y shape: {Y_arr.shape}")

# Split into train and validation (90/10 for monitoring only, not for early stopping)
val_split = 0.1
n_total = X_arr.shape[0]
n_val = int(n_total * val_split)
n_train = n_total - n_val

X_train, Y_train = X_arr[:n_train], Y_arr[:n_train]
X_val, Y_val = X_arr[n_train:], Y_arr[n_train:]

print(f"  Train samples: {n_train}, Val samples: {n_val}")

# Create DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(Y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(Y_val, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Create model
print(f"Creating {args.model_type.upper()} model...")
n_features = X_arr.shape[2]

if args.model_type == 'gru':
    model = GRUForecaster(
        input_size=X_arr.shape[1],  # number of features per timestep
        hidden_size=args.gru_d_model,
        num_layers=args.gru_n_layers,
        output_size=1,
        horizon=args.horizon,
        dropout=args.gru_dropout
    ).to(device)
elif args.model_type == 'tcn':
    model = TCNModel(
        input_channels=X_arr.shape[1],  # sequence length
        output_horizon=args.horizon,
        num_targets=1,
        hidden_channels=args.tcn_hidden_channels,
        levels=args.tcn_levels,
        kernel_size=args.tcn_kernel_size,
        dilation_base=2,
        dropout=args.tcn_dropout
    ).to(device)
elif args.model_type == 'patchtst':
    model = PatchTSTModel(
        input_channels=X_arr.shape[1],  # sequence length
        output_horizon=args.horizon,
        num_targets=1,
        context_length=args.input_length,
        d_model=args.patchtst_d_model,
        n_heads=args.patchtst_n_heads,
        n_layers=args.patchtst_n_layers,
        patch_len=args.patchtst_patch_len,
        patch_stride=args.patchtst_stride,
        d_ff=4*args.patchtst_d_model,
        dropout=args.patchtst_dropout
    ).to(device)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training loop
print("\nStarting training...")
print("=" * 80)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    
    # Training
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        
        # Handle different output shapes
        if Y_pred.dim() == 3 and Y_pred.size(1) == 1:
            Y_pred = Y_pred.squeeze(1)
        
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= n_train
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            Y_pred = model(X_batch)
            if Y_pred.dim() == 3 and Y_pred.size(1) == 1:
                Y_pred = Y_pred.squeeze(1)
            
            loss = criterion(Y_pred, Y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= n_val
    val_losses.append(val_loss)
    
    epoch_time = time.time() - epoch_start_time
    
    # Logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
    
    # Early stopping check
    if val_loss < best_val_loss - args.early_stopping_eps:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(OUTPUT_DIR, 'best_model.pt'))
    else:
        patience_counter += 1
    
    if patience_counter >= args.early_stopping_patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# Save final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}, os.path.join(OUTPUT_DIR, 'final_model.pt'))

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss,
    'epochs_trained': epoch + 1
}
with open(os.path.join(OUTPUT_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history, f)

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'{args.model_type.upper()} Training on Long Window\n{args.region} - {args.feature_set}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

writer.close()

print("\n" + "=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"Best Val Loss: {best_val_loss:.6f}")
print(f"Epochs Trained: {epoch + 1}")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 80)
