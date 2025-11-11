import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'forecast', 'multi_region', 'patchtst_multi_train')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import random, math, time
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import joblib
import pickle

from utils.forecast.set_seed import set_seed
from utils.forecast.construct_seq import construct_sequences_multi_output, preview_total_plot
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars
from utils.data_utils.info_cities import list_multi1, list_multi2, list_multi3, list_multi4, list_multi5, list_multi6, list_multi7 # options for training
from utils.data_utils.info_features import list_datetime, list_iseo_vars, list_F0, list_F1, list_F2, list_F3

from utils.forecast.PatchTST import PatchTSTModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# arg parse
import argparse
parser = argparse.ArgumentParser(description='PatchTST per-region single training experiment')

# random seed
parser.add_argument('--seed', type=int, default=597, help='random seed')
# choose a list as the training region
parser.add_argument('--region_list', type=int, default='1', help='which multi-region list to use for training: 1 to 7')

# weather features to use (type of experiments: F0-only ieso electricity, F1-all meteo+ieso electricity, F2-non-causally selected meteo+ieso electricity, F3-causally selected meteo+ieso electricity)
parser.add_argument('--feature_set', type=str, default='F3', help='feature set to use', choices=['F0', 'F1', 'F2', 'F3'])
# data directory
parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data', 'ieso_era5'), help='data directory path')
parser.add_argument('--data_file_prefix', type=str, default='combined_ieso_era5_avg_', help='data file prefix')

# scalers
parser.add_argument('--scaler', type=str, default='standard', choices=['minmax', 'standard', 'none'], help='scaling method')

# forecast parameters
parser.add_argument("--n_folds", type=int, default=5, help="Number of folds/window splits for time series cross validation")
parser.add_argument("--window_size", type=int, default=24*2*366, help="Window size for each fold (in hours) - will be the total train+test of this fold")
parser.add_argument("--train_ratio", type=float, default=0.93, help="Training (train+val) set ratio per fold")
parser.add_argument("--input_length", type=int, default=168, help="Input sequence length L (e.g., 168 for 1 week)")
parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon H (e.g., 24 hours)")
parser.add_argument("--stride", type=int, default=1, help="Stride between prediction windows")
parser.add_argument("--aggregation_mode", type=str, choices=["mean", "first"], default="mean", help="Overlap aggregation method")

# Training parameters
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--val_ratio", type=float, default=0.07, help="Fraction of data for validation")
parser.add_argument("--early_stopping_eps", type=float, default=1e-4, help="Minimum validation loss improvement for reset patience")
parser.add_argument("--early_stopping_patience", type=int, default=20, help="Epochs to wait without improvement before stopping")
parser.add_argument("--val_warmup", type=int, default=5, help="Epoch to start validation loss tracking for early stopping")

# PatchTST model hyperparameters
parser.add_argument("--d_model", type=int, default=32, help="Transformer model dimension")
parser.add_argument("--n_heads", type=int, default=4, help="Number of self-attention heads")
parser.add_argument("--n_layers", type=int, default=3, help="Number of Transformer encoder layers")
parser.add_argument("--patch_len", type=int, default=16, help="Patch length")
parser.add_argument("--patch_stride", type=int, default=8, help="Patch stride")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

args = parser.parse_args()

# Unpack arguments
seed = args.seed
set_seed(seed)

region_list_option = args.region_list
feature_set = args.feature_set


DATA_DIR = args.data_dir
DATA_FILE_PREFIX = args.data_file_prefix

scaler = args.scaler

n_folds = args.n_folds
window_size = args.window_size
train_ratio = args.train_ratio
input_length = args.input_length
horizon = args.horizon
stride = args.stride
aggregation_mode = args.aggregation_mode

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
val_ratio = args.val_ratio
early_stopping_eps = args.early_stopping_eps
early_stopping_patience = args.early_stopping_patience
val_warmup = args.val_warmup

d_model = args.d_model
n_heads = args.n_heads
n_layers = args.n_layers
patch_len = args.patch_len
patch_stride = args.patch_stride
dropout = args.dropout

# set up save path with hyperparameters and related info
OUTPUT_DIR=os.path.join(OUTPUT_DIR, f'region_list{region_list_option}')

exp_name = f"dm{d_model}_nh{n_heads}_nl{n_layers}_pl{patch_len}_ps{patch_stride}_inlen{input_length}_h{horizon}_scaler{scaler}"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, exp_name)

if stride==horizon:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'non_overlap')
else:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'overlap', f'stride{stride}', aggregation_mode)
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_info=f"bs{batch_size}_ep{epochs}_lr{lr}_tr{train_ratio}_vr{val_ratio}_pat{early_stopping_patience}_esep{early_stopping_eps}"

OUTPUT_DIR = os.path.join(OUTPUT_DIR, training_info, feature_set, str(seed))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# tensorboard writer
TB_DIR = os.path.join(OUTPUT_DIR, 'tensorboard_runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter(log_dir=TB_DIR)

# load data - city/region
# load the df for each city/region in the list, then combine with each column renamed to {city/region}_{var}
if region_list_option==1:
    region_list = list_multi1
elif region_list_option==2:
    region_list = list_multi2
elif region_list_option==3:
    region_list = list_multi3
elif region_list_option==4:
    region_list = list_multi4
elif region_list_option==5:
    region_list = list_multi5
elif region_list_option==6:
    region_list = list_multi6
elif region_list_option==7:
    region_list = list_multi7
else:
    raise ValueError("region_list_option must be an integer from 1 to 7")

# select features based on feature_set
if feature_set=='F0':
    selected_features = list_F0
elif feature_set=='F1':
    selected_features = list_F1
elif feature_set=='F2':
    selected_features = list_F2
elif feature_set=='F3':
    selected_features = list_F3
else:
    raise ValueError(f'Unknown feature set: {feature_set}')

data_frames = []
for region in region_list:
    print(f'Loading data for region/city: {region}')
    if region in dict_regions:
        if len(dict_regions[region])>1: # need to do each city and then average (since it's meteo variables, cannot aggregate)
            list_tmp_region_dfs = []
            for city in dict_regions[region]:
                print(f'  Loading city: {city}')
                data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{city.replace(" ", "_").lower()}.csv')
                city_df = pd.read_csv(data_file, parse_dates=['time'])
                city_df = city_df[['time'] + list_vars]
                city_df.set_index('time', inplace=True)
                city_df = city_df.rename(columns={var: f'{region}_{city}_{var}' for var in list_vars})
                list_tmp_region_dfs.append(city_df)
            # average over cities to get the region-level meteo reading
            df_region = pd.concat(list_tmp_region_dfs, axis=1)
            # aggregate/take average the same variables across cities - the name of col is {region}_{city}_{var_name}
            # for total consumption and premise count, we sum them, the rest we average them
            list_agg_dfs = []
            for var in list_vars:
                cols_var = [col for col in df_region.columns if col.endswith(f'_{var}')]
                if var in list_ieso_vars: # total consumption and premise count - sum
                    df_var = df_region[cols_var].sum(axis=1).to_frame(name=f'{region}_{var}')
                else: # meteo variables - average
                    df_var = df_region[cols_var].mean(axis=1).to_frame(name=f'{region}_{var}')
                list_agg_dfs.append(df_var)
            df_region = pd.concat(list_agg_dfs, axis=1)
            
        else: # only one city in the region
            data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{dict_regions[region][0].replace(" ", "_").lower()}.csv')
            df_region = pd.read_csv(data_file, parse_dates=['time'])
            df_region = df_region[['time'] + list_vars]
            df_region.set_index('time', inplace=True)
            df_region = df_region.rename(columns={var: f'{region}_{var}' for var in list_vars})
    else: 
        if region in list_cities: # single city within a region
            print(f'  Loading city: {region}')
            data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{region.replace(" ", "_").lower()}.csv')
            city_df = pd.read_csv(data_file, parse_dates=['time'])
            city_df = city_df[['time'] + list_vars]
            city_df.set_index('time', inplace=True)
            city_df = city_df.rename(columns={var: f'{region}_{var}' for var in list_vars})
            df_region = city_df
        else:
            raise ValueError(f'Unknown region/city name: {region}')
        
    # drop "AVG_CONSUMPTION_PER_PREMISE" since it's derived from other two variables
    df_region = df_region.drop(columns=[col for col in df_region.columns if col.endswith('AVG_CONSUMPTION_PER_PREMISE')])
    # select only the selected features
    selected_cols = []
    for feat in selected_features:
        if feat in list_ieso_vars:
            selected_cols += [col for col in df_region.columns if col.endswith(f'_{feat}')]
        else:
            selected_cols += [col for col in df_region.columns if col.endswith(f'_{feat}')]
    df_region = df_region[selected_cols]

    data_frames.append(df_region)

# combine all regions/cities dataframes with selected feature 
df_region = pd.concat(data_frames, axis=1)
df_region = df_region.sort_index()
df_region = df_region.astype('float32')

# add time features
df_region = time_features(df_region.reset_index(), time_col='time', drop_original=False).set_index('time')

# depending on the n_folds, window_size, and actual length of the dataframe, determine the splits (allow overlap if needed)
total_length = df_region.shape[0]
fold_windows = []
max_start_index = total_length - window_size
if n_folds*window_size <= total_length:
    # no overlap needed
    for fold in range(n_folds):
        start_idx = fold * window_size
        end_idx = start_idx + window_size
        fold_windows.append( (start_idx, end_idx) )
else:
    # overlap needed
    if n_folds==1:
        fold_windows.append( (0, window_size) )
    else:
        overlap_size = (n_folds * window_size - total_length) // (n_folds - 1)
        step_size = window_size - overlap_size
        for fold in range(n_folds):
            start_idx = fold * step_size
            end_idx = start_idx + window_size
            if end_idx > total_length:
                end_idx = total_length
                start_idx = end_idx - window_size
            fold_windows.append( (start_idx, end_idx) )


# to track: the losses of each fold, the average per epoch training time per fold
fold_train_losses = []
fold_val_losses = []

fold_test_maes = []
fold_test_rmses = []
fold_test_mapes = []
fold_test_smapes = []
fold_test_mases=[]

fold_epoch_times = []


# print fold windows info
for fold, (start_idx, end_idx) in enumerate(fold_windows):
    start_time = df_region.index[start_idx]
    end_time = df_region.index[end_idx - 1]
    print(f'Fold {fold}: Index range ({start_idx}, {end_idx}) --> Time range ({start_time} to {end_time})')

    OUTPUT_DIR_FOLD = os.path.join(OUTPUT_DIR, f'fold_{fold}') # save the results for each, then in upper level save the average
    os.makedirs(OUTPUT_DIR_FOLD, exist_ok=True)

    # save fold window info
    with open(os.path.join(OUTPUT_DIR_FOLD, 'fold_window_info.txt'), 'w') as f:
        f.write(f'Fold {fold}: Index range ({start_idx}, {end_idx}) --> Time range ({start_time} to {end_time})\n')

    # get the fold data
    df_fold = df_region.iloc[start_idx:end_idx].copy()
    # the train_eval_test split indices
    n_total = df_fold.shape[0]
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    n_val = int(n_train * val_ratio)
    n_train_final = n_train - n_val 
    train_df_fold = df_fold.iloc[:n_train_final]
    val_df_fold = df_fold.iloc[n_train_final:n_train]
    test_df_fold = df_fold.iloc[n_train:]
    print(f'  Train/Val/Test sizes: {train_df_fold.shape[0]}/{val_df_fold.shape[0]}/{test_df_fold.shape[0]}')


    # scale the training data (if scaler is not none)
    if scaler.lower()!='none' and scaler is not None:
        # fit scaler on training data
        if scaler.lower()=='minmax':
            scaler_all = MinMaxScaler()
        elif scaler.lower()=='standard':
            scaler_all = StandardScaler()
        else:
            raise ValueError(f'Unknown scaling method: {scaler}')
        
        train_df_fold = train_df_fold.astype('float32')
        # fit on values to avoid sklearn "feature names" warnings when transforming numpy arrays
        scaler_all.fit(train_df_fold.values)
        # save the scaler
        with open(os.path.join(OUTPUT_DIR_FOLD, f'scaler_all.pkl'), 'wb') as f:
            pickle.dump(scaler_all, f)
        # transform training data
        train_df_fold.loc[:, :] = scaler_all.transform(train_df_fold.values)

    # construct training sequences
    X_arr, Y_arr = construct_sequences_multi_output(train_df_fold, input_length, horizon, step_size=stride, target_cols=[f'{region}_TOTAL_CONSUMPTION' for region in region_list])
    print(f'  Training sequences: X shape {X_arr.shape}, Y shape {Y_arr.shape}')

    # prepare fixed evaluation snapshot - get truth unnormalized values
    eval_snap_df = val_df_fold.copy()
    
    # Store unnormalized truth BEFORE scaling
    truth_eval_snap_unnorm = {}
    for region in region_list:
        truth_eval_snap_unnorm[region] = eval_snap_df[f'{region}_TOTAL_CONSUMPTION'].values.copy()

    # scale it with the training scaler
    if scaler.lower()!='none' and scaler is not None:
        eval_snap_df.loc[:, :] = scaler_all.transform(eval_snap_df.values)

    # If validation set is too short to form at least one (L+H) window, fall back to the tail of the training set
    min_needed = input_length + horizon
    if eval_snap_df.shape[0] < min_needed:
        print(f"  Warning: validation snapshot too short ({eval_snap_df.shape[0]} rows) for L+H={min_needed}. Using tail of training data for eval snapshot.")
        # take the last min_needed rows from train_df_fold (already scaled)
        eval_snap_df = train_df_fold.iloc[-min_needed:].copy()
        # Also update unnormalized truth from the original (unscaled) training data
        # We need to get the original unscaled training data for this
        original_train_tail = df_fold.iloc[:n_train_final].iloc[-min_needed:]
        for region in region_list:
            truth_eval_snap_unnorm[region] = original_train_tail[f'{region}_TOTAL_CONSUMPTION'].values.copy()
    # build eval snapshot sequences (now eval_snap_df has at least L+H rows)
    X_snap_arr, Y_snap_arr = construct_sequences_multi_output(eval_snap_df, input_length, horizon, step_size=stride, target_cols=[f'{region}_TOTAL_CONSUMPTION' for region in region_list])
    print(f'  Eval snapshot sequences: X shape {X_snap_arr.shape}, Y shape {Y_snap_arr.shape}')

    # train and val loaders
    train_loader=DataLoader(TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(Y_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.tensor(X_snap_arr, dtype=torch.float32), torch.tensor(Y_snap_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=False)


    # initialize model
    model = PatchTSTModel(input_channels=X_arr.shape[1], output_horizon=horizon, num_targets=Y_arr.shape[1], 
                        context_length=input_length, d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
                        patch_len=patch_len, patch_stride=patch_stride, d_ff=4*d_model, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    trigger_times = 0
    global_step = 0
    model_file = "best_model.pth"

    train_losses = []
    val_losses = []

    epoch_times = [] # time spent on each epoch

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            global_step += 1

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # tensorboard logging
        writer.add_scalar(f'Fold_{fold}/Train_Loss', train_loss, epoch)
        writer.add_scalar(f'Fold_{fold}/Val_Loss', val_loss, epoch)

        print(f'    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_duration:.2f}s')

        # early stopping check
        if epoch < val_warmup: # save initial epochs without any checks
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR_FOLD, model_file))
            best_val_loss = val_loss
        else:
            if val_loss + early_stopping_eps < best_val_loss:
                print(f'    Validation loss improved, saving model... Improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model.')
                best_val_loss = val_loss
                torch.save(model.state_dict(),  os.path.join(OUTPUT_DIR_FOLD, model_file))
                trigger_times = 0
            else:
                trigger_times += 1
                print(f'    No improvement in validation loss for {trigger_times} epochs')
                if trigger_times >= early_stopping_patience:
                    print(f'    Early stopping at epoch {epoch}')
                    break


        # tensorboard logging
        writer.add_scalar(f'Fold_{fold}/Train_Loss', train_loss, epoch)
        writer.add_scalar(f'Fold_{fold}/Val_Loss', val_loss, epoch)
        print(f'  Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # log gradient norms occationally
        if epoch % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar(f'Fold_{fold}/Grad_Total_l2_Norm', total_norm, epoch)

        # preview on fixed eval snapshot every 10 epochs
        if (epoch+1) % 10 == 0:
            # For multi-output model, preview per region using the corresponding unnormalized truth and scaler mean/std
            for reg in region_list:
                # get mean/std for this region's TOTAL_CONSUMPTION from the fitted scaler using the dataframe column locations
                if scaler.lower()!='none' and scaler is not None:
                    try:
                        col_idx = train_df_fold.columns.get_loc(f'{reg}_TOTAL_CONSUMPTION')
                        load_mean = float(scaler_all.mean_[col_idx])
                        load_std = float(scaler_all.scale_[col_idx])
                    except Exception:
                        # Fallback: if column not found, use 0/1
                        load_mean = 0.0
                        load_std = 1.0
                else:
                    load_mean = 0.0
                    load_std = 1.0

                # truth for this region (unnormalized)
                truth_reg = truth_eval_snap_unnorm[reg]

                # preview only the first target channel (preview_total_plot expects 1-D truth)
                target_idx = region_list.index(reg)
                preview_total_plot(model, val_loader, eval_snap_df.index.values, truth_reg, eval_snap_df.shape[0], input_length, horizon, stride, aggregation_mode,
                                    load_mean=load_mean, load_std=load_std,
                                    save_dir=OUTPUT_DIR_FOLD, device=device, writer=writer, step=epoch+1, target_channel=target_idx)
            
    # end of epoch loop
    # log fold train and val losses
    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)
    fold_epoch_times.append(epoch_times)
    print(f'  Fold {fold} training completed. Average epoch time: {np.mean(epoch_times):.2f} seconds.')
    # save avg epoch time
    with open(os.path.join(OUTPUT_DIR_FOLD, 'epoch_times.txt'), 'w') as f:
        for epoch_idx, epoch_time in enumerate(epoch_times):
            f.write(f'Epoch {epoch_idx}: {epoch_time:.4f} seconds\n')
        f.write(f'Average epoch time: {np.mean(epoch_times):.4f} seconds\n')


    # now evaluate on test set using the best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR_FOLD, model_file), map_location=device))
    model.eval()
    # construct test sequences
    test_df_fold_original = test_df_fold.copy()
    test_df_fold = test_df_fold.astype('float32')
    if scaler.lower()!='none' and scaler is not None:
        test_df_fold.loc[:, :] = scaler_all.transform(test_df_fold.values)
    # if test set too short to form any (L+H) window, skip test evaluation for this fold
    min_needed = input_length + horizon
    if test_df_fold.shape[0] < min_needed:
        print(f"  Warning: test set too short ({test_df_fold.shape[0]} rows) for L+H={min_needed}. Skipping test evaluation for this fold.")
        # record NaNs lists for metrics (one per region) to keep lengths consistent
        nan_list = [np.nan] * len(region_list)
        fold_test_maes.append(nan_list)
        fold_test_rmses.append(nan_list)
        fold_test_mapes.append(nan_list)
        fold_test_smapes.append(nan_list)
        fold_test_mases.append(nan_list)
        # skip to next fold
        continue
    X_test_arr, Y_test_arr = construct_sequences_multi_output(test_df_fold, input_length, horizon, step_size=stride, target_cols=[f'{region}_TOTAL_CONSUMPTION' for region in region_list])
    print(f'  Test sequences: X shape {X_test_arr.shape}, Y shape {Y_test_arr.shape}')
    test_loader=DataLoader(TensorDataset(torch.tensor(X_test_arr, dtype=torch.float32), torch.tensor(Y_test_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    # get predictions
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb) # (
            all_preds.append(out.cpu().numpy())
            all_truths.append(yb.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_truths = np.concatenate(all_truths, axis=0)

    # aggregate overlaps per time step
    t_starts=np.arange(0, n_test - input_length - horizon + 1, stride)
    if len(t_starts) != all_preds.shape[0]:
        t_starts = np.arange((all_preds.shape[0]))
    bags = defaultdict(list)   # t_idx -> list of (C,)
    for w, t0 in enumerate(t_starts):
        for h in range(0,horizon):
            ti = t0 + input_length + h
            if ti >= n_test: break
            bags[ti].append(all_preds[w, :, h])
    pred_idx = sorted(bags.keys())
    if aggregation_mode=="mean":
         agg_pred_norm = np.array([np.mean(bags[t], axis=0) for t in pred_idx])
    elif aggregation_mode=="first":
         agg_pred_norm = np.array([bags[t][0] for t in pred_idx])
    else:
        raise ValueError(f'Unknown aggregation mode: {aggregation_mode}')
    
    # denormalize predictions and truths - note that the out put is multi-output for multiple regions
    agg_pred = np.zeros_like(agg_pred_norm)
    agg_truth = np.zeros_like(agg_pred_norm)
    for i, region in enumerate(region_list):
        if scaler.lower()!='none' and scaler is not None:
            try:
                col_idx = train_df_fold.columns.get_loc(f'{region}_TOTAL_CONSUMPTION')
                mean_tc = float(scaler_all.mean_[col_idx])
                std_tc = float(scaler_all.scale_[col_idx])
            except Exception:
                # fallback if column not found in training df (shouldn't happen), use overall defaults
                mean_tc = float(scaler_all.mean_.mean() if hasattr(scaler_all, 'mean_') else 0.0)
                std_tc = float(np.mean(scaler_all.scale_) if hasattr(scaler_all, 'scale_') else 1.0)
            agg_pred[:, i] = agg_pred_norm[:, i] * std_tc + mean_tc
            # get the corresponding truths from original (unnormalized) test df
            truth_region = test_df_fold_original[f'{region}_TOTAL_CONSUMPTION'].values
            # construct the aggregated truth based on t_starts and aggregation mode
            bags_truth = defaultdict(list)
            for w, t0 in enumerate(t_starts):
                for h in range(0,horizon):
                    ti = t0 + input_length + h
                    if ti >= n_test: break
                    bags_truth[ti].append(truth_region[ti])
            if aggregation_mode=="mean":
                agg_truth[:, i] = np.array([np.mean(bags_truth[t]) for t in pred_idx])
            elif aggregation_mode=="first":
                agg_truth[:, i] = np.array([bags_truth[t][0] for t in pred_idx])
        else:
            agg_pred[:, i] = agg_pred_norm[:, i]
            # get the corresponding truths from original (unnormalized) test df
            truth_region = test_df_fold_original[f'{region}_TOTAL_CONSUMPTION'].values
            # construct the aggregated truth based on t_starts and aggregation mode
            bags_truth = defaultdict(list)
            for w, t0 in enumerate(t_starts):
                for h in range(0,horizon):
                    ti = t0 + input_length + h
                    if ti >= n_test: break
                    bags_truth[ti].append(truth_region[ti])
            if aggregation_mode=="mean":
                agg_truth[:, i] = np.array([np.mean(bags_truth[t]) for t in pred_idx])
            elif aggregation_mode=="first":
                agg_truth[:, i] = np.array([bags_truth[t][0] for t in pred_idx])

    # align with true values
    aligned_dt = test_df_fold_original.index[pred_idx]
    true_aligned = test_df_fold_original[[f'{region}_TOTAL_CONSUMPTION' for region in region_list]].values[pred_idx, :]

    # save predictions to csv
    results_df=pd.DataFrame({"datetime": aligned_dt, **{f"predicted_{region}": agg_pred[:, i] for i, region in enumerate(region_list)}, **{f'true_{region}': true_aligned[:, i] for i, region in enumerate(region_list)}})
    results_df.to_csv(os.path.join(OUTPUT_DIR_FOLD, 'test_predictions.csv'), index=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR_FOLD, 'test_predictions.csv'), index=False)

    # compute metrics per region
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    list_smape=[]
    list_mase=[]
    for i, region in enumerate(region_list):
        preds_region = agg_pred[:, i]
        truths_region = agg_truth[:, i]
        mae = np.mean(np.abs(preds_region - truths_region))
        rmse = np.sqrt(np.mean((preds_region - truths_region)**2))
        mape = np.mean( np.abs((preds_region - truths_region) / (truths_region + 1e-8)) ) * 100.0
        smape = np.mean( 2.0 * np.abs(preds_region - truths_region) / (np.abs(preds_region) + np.abs(truths_region) + 1e-8) ) * 100.0
        # MASE
        # naive seasonal forecast using seasonality of 24 (daily)
        seasonality = 24
        n = len(truths_region)
        d = np.sum( np.abs(truths_region[seasonality:] - truths_region[:-seasonality]) ) / (n - seasonality)
        errors = np.abs(truths_region - preds_region)
        mase = np.mean(errors) / (d + 1e-8)

        list_mae.append(mae)
        list_rmse.append(rmse)
        list_mape.append(mape)
        list_smape.append(smape)
        list_mase.append(mase)

        with open(os.path.join(OUTPUT_DIR_FOLD, f'test_metrics_{region}.txt'), 'w') as f:
            f.write(f'Test Metrics for region {region}:\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MAPE: {mape:.4f}%\n')
            f.write(f'sMAPE: {smape:.4f}%\n')
            f.write(f'MASE: {mase:.4f}\n')
        print(f'    Test metrics for region {region} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%, sMAPE: {smape:.4f}%, MASE: {mase:.4f}')

    # append per-region metrics for this fold
    fold_test_maes.append(list_mae)
    fold_test_rmses.append(list_rmse)
    fold_test_mapes.append(list_mape)
    fold_test_smapes.append(list_smape)
    fold_test_mases.append(list_mase)

    # Plot the rolling forecast vs truth per region and save
    for i, region in enumerate(region_list):
        plt.figure(figsize=(12,6))
        plt.plot(aligned_dt, true_aligned[:, i], label="True Load", linestyle="--")
        plt.plot(aligned_dt, agg_pred[:, i], label="Predicted Load", marker='o')
        plt.title(f"Total Load Forecast for {region} (H={horizon}, stride={stride}, agg={aggregation_mode})\nMAE={list_mae[i]:.4f}, RMSE={list_rmse[i]:.4f}")
        plt.xlabel("Datetime"); plt.ylabel("Load"); plt.xticks(rotation=45)
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_FOLD, f'test_forecast_{region}.png'))
        plt.close()
    
        # Write overall results across folds per region
        with open(os.path.join(OUTPUT_DIR, 'overall_results.txt'), 'w') as f:
            f.write(f'Overall results across {n_folds} folds for regions: {region_list}\n')
            f.write('\n')
            # For each metric, write a table of folds x regions
            f.write('MAE per fold (rows: fold, cols: region)\n')
            f.write('\t' + '\t'.join(region_list) + '\n')
            for fold_idx in range(len(fold_test_maes)):
                row = []
                for reg_idx in range(len(region_list)):
                    try:
                        val = fold_test_maes[fold_idx][reg_idx]
                        row.append(f"{val:.4f}" if not np.isnan(val) else 'nan')
                    except Exception:
                        row.append('nan')
                f.write(f'Fold_{fold_idx}\t' + '\t'.join(row) + '\n')
            # average MAE per region
            f.write('Average MAE per region\n')
            avg_row = []
            for reg_idx in range(len(region_list)):
                vals = [fold_test_maes[fi][reg_idx] for fi in range(len(fold_test_maes)) if not np.isnan(fold_test_maes[fi][reg_idx])]
                avg = np.mean(vals) if len(vals)>0 else np.nan
                avg_row.append(f"{avg:.4f}" if not np.isnan(avg) else 'nan')
            f.write('\t' + '\t'.join(avg_row) + '\n\n')
    
            # RMSE
            f.write('RMSE per fold (rows: fold, cols: region)\n')
            f.write('\t' + '\t'.join(region_list) + '\n')
            for fold_idx in range(len(fold_test_rmses)):
                row = []
                for reg_idx in range(len(region_list)):
                    try:
                        val = fold_test_rmses[fold_idx][reg_idx]
                        row.append(f"{val:.4f}" if not np.isnan(val) else 'nan')
                    except Exception:
                        row.append('nan')
                f.write(f'Fold_{fold_idx}\t' + '\t'.join(row) + '\n')
            f.write('Average RMSE per region\n')
            avg_row = []
            for reg_idx in range(len(region_list)):
                vals = [fold_test_rmses[fi][reg_idx] for fi in range(len(fold_test_rmses)) if not np.isnan(fold_test_rmses[fi][reg_idx])]
                avg = np.mean(vals) if len(vals)>0 else np.nan
                avg_row.append(f"{avg:.4f}" if not np.isnan(avg) else 'nan')
            f.write('\t' + '\t'.join(avg_row) + '\n')
    
            # MAPE per fold
            f.write('\nMAPE per fold (rows: fold, cols: region)\n')
            f.write('\t' + '\t'.join(region_list) + '\n')
            for fold_idx in range(len(fold_test_mapes)):
                row = []
                for reg_idx in range(len(region_list)):
                    try:
                        val = fold_test_mapes[fold_idx][reg_idx]
                        row.append(f"{val:.4f}" if not np.isnan(val) else 'nan')
                    except Exception:
                        row.append('nan')
                f.write(f'Fold_{fold_idx}\t' + '\t'.join(row) + '\n')
            f.write('Average MAPE per region\n')
            avg_row = []
            for reg_idx in range(len(region_list)):
                vals = [fold_test_mapes[fi][reg_idx] for fi in range(len(fold_test_mapes)) if not np.isnan(fold_test_mapes[fi][reg_idx])]
                avg = np.mean(vals) if len(vals)>0 else np.nan
                avg_row.append(f"{avg:.4f}" if not np.isnan(avg) else 'nan')
            f.write('\t' + '\t'.join(avg_row) + '\n')

            # sMAPE per fold
            f.write('\nSMAPE per fold (rows: fold, cols: region)\n')
            f.write('\t' + '\t'.join(region_list) + '\n')
            for fold_idx in range(len(fold_test_smapes)):
                row = []
                for reg_idx in range(len(region_list)):
                    try:
                        val = fold_test_smapes[fold_idx][reg_idx]
                        row.append(f"{val:.4f}" if not np.isnan(val) else 'nan')
                    except Exception:
                        row.append('nan')
                f.write(f'Fold_{fold_idx}\t' + '\t'.join(row) + '\n')
            f.write('Average SMAPE per region\n')
            avg_row = []
            for reg_idx in range(len(region_list)):
                vals = [fold_test_smapes[fi][reg_idx] for fi in range(len(fold_test_smapes)) if not np.isnan(fold_test_smapes[fi][reg_idx])]
                avg = np.mean(vals) if len(vals)>0 else np.nan
                avg_row.append(f"{avg:.4f}" if not np.isnan(avg) else 'nan')
            f.write('\t' + '\t'.join(avg_row) + '\n')

            # MASE per fold
            f.write('\nMASE per fold (rows: fold, cols: region)\n')
            f.write('\t' + '\t'.join(region_list) + '\n')
            for fold_idx in range(len(fold_test_mases)):
                row = []
                for reg_idx in range(len(region_list)):
                    try:
                        val = fold_test_mases[fold_idx][reg_idx]
                        row.append(f"{val:.4f}" if not np.isnan(val) else 'nan')
                    except Exception:
                        row.append('nan')
                f.write(f'Fold_{fold_idx}\t' + '\t'.join(row) + '\n')
            f.write('Average MASE per region\n')
            avg_row = []
            for reg_idx in range(len(region_list)):
                vals = [fold_test_mases[fi][reg_idx] for fi in range(len(fold_test_mases)) if not np.isnan(fold_test_mases[fi][reg_idx])]
                avg = np.mean(vals) if len(vals)>0 else np.nan
                avg_row.append(f"{avg:.4f}" if not np.isnan(avg) else 'nan')
            f.write('\t' + '\t'.join(avg_row) + '\n')
    
        print(f'Experiment completed. Overall results saved to {os.path.join(OUTPUT_DIR, "overall_results.txt")}')

