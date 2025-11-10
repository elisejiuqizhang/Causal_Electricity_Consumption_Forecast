# load the data of one city/region
# specify the weather features to use (non, all, or selected set)
# train and evaluate the model to forecast the electricity consumption of that city/region
# Time series cross validation (sliding window) is used
# Results are saved per region

import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'forecast', 'per_region', 'patchtst_single_train')
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
from utils.forecast.construct_seq import construct_sequences_uni_output, preview_total_plot
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars
from utils.data_utils.info_features import list_datetime, list_iseo_vars, list_F0, list_F1, list_F2, list_F3

from utils.forecast.PatchTST import PatchTSTModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# arg parse
import argparse
parser = argparse.ArgumentParser(description='PatchTST per-region single training experiment')

# random seed
parser.add_argument('--seed', type=int, default=597, help='random seed')
# city/region name
parser.add_argument('--region', type=str, default='Peel', help='name of the region to run experiment on', choices=list(dict_regions.keys())+list_cities)
# weather features to use (type of experiments: F0-only ieso electricity, F1-all meteo+ieso electricity, F2-non-causally selected meteo+ieso electricity, F3-causally selected meteo+ieso electricity)
parser.add_argument('--feature_set', type=str, default='F2', help='feature set to use', choices=['F0', 'F1', 'F2', 'F3'])
# data directory
parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data', 'ieso_era5'), help='data directory path')
parser.add_argument('--data_file_prefix', type=str, default='combined_ieso_era5_avg_', help='data file prefix')

# scalers
parser.add_argument('--scaler', type=str, default='standard', choices=['minmax', 'standard', 'none'], help='scaling method')

# forecast parameters
parser.add_argument("--n_folds", type=int, default=3, help="Number of folds/window splits for time series cross validation")
parser.add_argument("--window_size", type=int, default=24*2*366, help="Window size for each fold (in hours) - will be the total train+test of this fold")
parser.add_argument("--train_ratio", type=float, default=0.93, help="Training (train+val) set ratio per fold")
parser.add_argument("--input_length", type=int, default=168, help="Input sequence length L (e.g., 168 for 1 week)")
parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon H (e.g., 24 hours)")
parser.add_argument("--stride", type=int, default=1, help="Stride between prediction windows")
parser.add_argument("--aggregation_mode", type=str, choices=["mean", "first"], default="mean", help="Overlap aggregation method")

# Training parameters
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
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

region = args.region
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
OUTPUT_DIR=os.path.join(OUTPUT_DIR, f'{region.replace(" ", "_")}')

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
        # df_region.groupby(lambda x: '_'.join(x.split('_')[2:]), axis=1)

        # rename - drop the {region}_ prefix to simplify
        df_region = df_region.rename(columns={col: '_'.join(col.split('_')[1:]) for col in df_region.columns})  



    else: # only one city in the region
        data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{dict_regions[region][0].replace(" ", "_").lower()}.csv')
        df_region = pd.read_csv(data_file, parse_dates=['time'])
        df_region = df_region[['time'] + list_vars]
        df_region.set_index('time', inplace=True)
        # df_region = df_region.rename(columns={var: f'{region}_{var}' for var in list_vars})
else: 
    if region in list_cities: # single city within a region
        print(f'  Loading city: {region}')
        data_file = os.path.join(DATA_DIR, f'{DATA_FILE_PREFIX}{region.replace(" ", "_").lower()}.csv')
        city_df = pd.read_csv(data_file, parse_dates=['time'])
        city_df = city_df[['time'] + list_vars]
        city_df.set_index('time', inplace=True)
        # city_df = city_df.rename(columns={var: f'{region}_{var}' for var in list_vars})
        df_region = city_df
    else:
        raise ValueError(f'Unknown region/city name: {region}')

# drop AVG_CONSUMPTION_PER_PREMISE
df_region = df_region.drop(columns=['AVG_CONSUMPTION_PER_PREMISE'], errors='ignore')
# add datetime features
df_region = time_features(df_region.reset_index(), time_col='time', drop_original=False).set_index('time')


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
df_region = df_region[selected_features]


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
        scaler_all.fit(train_df_fold.values)
        # save the scaler
        with open(os.path.join(OUTPUT_DIR_FOLD, f'scaler_all.pkl'), 'wb') as f:
            pickle.dump(scaler_all, f)
        # transform training data
        train_df_fold.loc[:, :] = scaler_all.transform(train_df_fold.values)


    # construct training sequences
    X_arr, Y_arr = construct_sequences_uni_output(train_df_fold, history_len=input_length, horizon=horizon, step_size=stride, target_col='TOTAL_CONSUMPTION')
    print(f'  Training sequences shape: X: {X_arr.shape}, Y: {Y_arr.shape}')


    # prepare a fixed evaluation snapshot - get truth unnormalized values
    eval_snap_df = val_df_fold.copy()
    truth_eval_snap = eval_snap_df['TOTAL_CONSUMPTION'].values
        
    # normalize the snapshot using the training 
    eval_snap_df = eval_snap_df.astype('float32')
    if scaler.lower()!='none' and scaler is not None:
        eval_snap_df.loc[:, :] = scaler_all.transform(eval_snap_df.values)
    # build eval snapshot sequences
    X_snap_arr, Y_snap_arr = construct_sequences_uni_output(eval_snap_df, history_len=input_length, horizon=horizon, step_size=stride, target_col='TOTAL_CONSUMPTION')
    print(f'  Eval snapshot sequences shape: X: {X_snap_arr.shape}, Y: {Y_snap_arr.shape}')

    # get train and val loaders
    train_loader=DataLoader(TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(Y_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.tensor(X_snap_arr, dtype=torch.float32), torch.tensor(Y_snap_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    # Initialize PatchTST model
    model = PatchTSTModel(input_channels=X_arr.shape[1], output_horizon=horizon, num_targets=1, 
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
        start_epoch_time = time.time()
        model.train()
        train_loss=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)  # (B, 1, H)
            # out = out[:, 0, :]  # (B, H)
            out=out.squeeze(1)  # (B, H)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            global_step += 1
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # evaluate validation losses - decide early stopping
        val_loss=0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)  # (B, 1, H)
                # out = out[:, 0, :]  # (B, H)
                out=out.squeeze(1)  # (B, H)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        epoch_times.append(time.time() - start_epoch_time)

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
            preview_total_plot(model, val_loader, eval_snap_df.index.values, truth_eval_snap, eval_snap_df.shape[0], input_length, horizon, stride, aggregation_mode, 
                                load_mean=scaler_all.mean_[selected_features.index('TOTAL_CONSUMPTION')] if scaler.lower()!='none' and scaler is not None else 0.0,
                                load_std=scaler_all.scale_[selected_features.index('TOTAL_CONSUMPTION')] if scaler.lower()!='none' and scaler is not None else 1.0,
                                save_dir=OUTPUT_DIR_FOLD, device=device, writer=writer, step=epoch+1)
            
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
    X_test_arr, Y_test_arr = construct_sequences_uni_output(test_df_fold, history_len=input_length, horizon=horizon, step_size=stride, target_col='TOTAL_CONSUMPTION')
    print(f'  Test sequences shape: X: {X_test_arr.shape}, Y: {Y_test_arr.shape}')
    test_loader=DataLoader(TensorDataset(torch.tensor(X_test_arr, dtype=torch.float32), torch.tensor(Y_test_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    # prepare rolling forecasts
    all_preds=[]
    all_tgts=[]
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)  # (B, 1, H)
            # out = out[:, 0, :]  # (B, H)
            out=out.squeeze(1)  # (B, H)
            all_preds.append(out.cpu().numpy())
            all_tgts.append(yb.numpy())
    all_preds=np.vstack(all_preds)  # (num_windows, H)
    all_tgts=np.vstack(all_tgts)    # (num_windows, H)
    num_windows, H_out = all_preds.shape
    # aggregate overlapping predictions across windows
    t_starts=np.arange(0, n_test - input_length - horizon + 1, stride)
    assert len(t_starts)==num_windows, f'Number of windows mismatch: {len(t_starts)} vs {num_windows}'
    ts_pred = defaultdict(list)
    for w, t0 in enumerate(t_starts):
        for h_step in range(horizon):
            t_idx = t0 + input_length + h_step
            if t_idx >= n_test: break
            ts_pred[t_idx].append(all_preds[w, h_step])
    pred_indices = sorted(ts_pred.keys())
    # Apply aggregation method
    if aggregation_mode == "mean":
        agg_pred_norm = np.array([np.mean(ts_pred[i]) for i in pred_indices], dtype=np.float32)
    elif aggregation_mode == "first":
        agg_pred_norm = np.array([ts_pred[i][0] for i in pred_indices], dtype=np.float32)
    else:
        raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
    # Denormalize predictions
    if scaler.lower()!='none' and scaler is not None:
        agg_pred = agg_pred_norm * scaler_all.scale_[selected_features.index('TOTAL_CONSUMPTION')] + scaler_all.mean_[selected_features.index('TOTAL_CONSUMPTION')]
    else:
        agg_pred = agg_pred_norm
    # align with true values
    aligned_dt=test_df_fold_original.index[pred_indices]
    true_aligned=test_df_fold_original['TOTAL_CONSUMPTION'].values[pred_indices]


    # save predictions to csv
    results_df = pd.DataFrame({"datetime": aligned_dt, "predicted_load": agg_pred, "true_load": true_aligned})
    results_df.to_csv(os.path.join(OUTPUT_DIR_FOLD, 'test_predictions.csv'), index=False)

    # compute overall MAE, MSE, mape, smape, mase on evaluation period
    if np.isfinite(true_aligned).all():
        diff = agg_pred - true_aligned
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff**2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean( np.abs(diff) / (np.abs(true_aligned) + 1e-8) )) * 100.0
        smape = float(np.mean( 2.0 * np.abs(diff) / (np.abs(agg_pred) + np.abs(true_aligned) + 1e-8) )) * 100.0
        # MASE
        naive_forecast = test_df_fold['TOTAL_CONSUMPTION'].values[input_length - horizon : n_test - horizon]
        mase_denominator = np.mean(np.abs(true_aligned - naive_forecast))
        mase = float(mae / (mase_denominator + 1e-8))
        print(f"  Test set results - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%, SMAPE: {smape:.4f}%, MASE: {mase:.4f}")

        fold_test_maes.append(mae)
        fold_test_rmses.append(rmse)
        fold_test_mapes.append(mape)
        fold_test_smapes.append(smape)
        fold_test_mases.append(mase)

        # Plot the rolling forecast vs truth
        plt.figure(figsize=(12,6))
        plt.plot(aligned_dt, true_aligned, label="True Load", linestyle="--")
        plt.plot(aligned_dt, agg_pred, label="Predicted Load", marker='o')
        plt.title(f"Total Load Forecast (H={horizon}, stride={stride}, agg={aggregation_mode})\nMAE={mae:.4f}, RMSE={rmse:.4f}")
        plt.xlabel("Datetime"); plt.ylabel("Load"); plt.xticks(rotation=45)
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_FOLD, 'test_forecast_plot.png'))
        plt.close()

# end of fold loop
# save overall results
with open(os.path.join(OUTPUT_DIR, 'overall_results.txt'), 'w') as f:
    f.write(f'Overall results across {n_folds} folds:\n')
    f.write(f'Fold\tTest_MAE\tTest_RMSE\n')
    for fold in range(n_folds):
        f.write(f'{fold}\t{fold_test_maes[fold]:.4f}\t{fold_test_rmses[fold]:.4f}\n')
    avg_mae = np.mean(fold_test_maes)
    avg_rmse = np.mean(fold_test_rmses)
    f.write(f'Average\t{avg_mae:.4f}\t{avg_rmse:.4f}\n')

    f.write(f'Fold\tTest_MAPE\tTest_SMAPE\tTest_MASE\n')
    for fold in range(n_folds):
        f.write(f'{fold}\t{fold_test_mapes[fold]:.4f}\t{fold_test_smapes[fold]:.4f}\t{fold_test_mases[fold]:.4f}\n')
    avg_mape = np.mean(fold_test_mapes)
    avg_smape = np.mean(fold_test_smapes)
    avg_mase = np.mean(fold_test_mases)
    f.write(f'Average\t{avg_mape:.4f}\t{avg_smape:.4f}\t{avg_mase:.4f}\n')
    
    f.write('\nAverage epoch times per fold:\n')
    for fold in range(n_folds):
        avg_epoch_time = np.mean(fold_epoch_times[fold])
        f.write(f'{fold}\t{avg_epoch_time:.4f} seconds\n')
    overall_avg_epoch_time = np.mean([np.mean(times) for times in fold_epoch_times])
    f.write(f'Average\t{overall_avg_epoch_time:.4f} seconds\n')
print(f'Experiment completed. Overall results saved to {os.path.join(OUTPUT_DIR, "overall_results.txt")}')
