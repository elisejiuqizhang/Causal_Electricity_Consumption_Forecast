import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def construct_sequences_uni_output(df, history_len, horizon, step_size=1, target_col='TOTAL_CONSUMPTION'):
    """
    Construct input-output sequences for time series forecasting - univariate output.

    Parameters:
    - df: pd.DataFrame, input time series data with datetime index
    - history_len: int, number of past time steps to use as input
    - horizon: int, number of future time steps to predict
    - step_size: int, step size between sequences

    Returns:
    - X: np.ndarray, input sequences of shape (num_samples, num_features, history_len)
    - Y: np.ndarray, output sequences of shape (num_samples, horizon)
    """
    data = df.values
    num_samples = (len(df) - history_len - horizon) // step_size + 1
    num_features = data.shape[1]

    X = np.zeros((num_samples, num_features, history_len))
    Y = np.zeros((num_samples, horizon))

    for i in range(num_samples):
        start_x = i * step_size
        end_x = start_x + history_len
        start_y = end_x
        end_y = start_y + horizon

        X[i] = data[start_x:end_x].T  # Transpose to get shape (num_features, history_len)
        Y[i] = data[start_y:end_y, df.columns.get_loc(target_col)]

    return X, Y

def construct_sequences_multi_output(df, history_len, horizon, step_size=1, target_cols=['TOTAL_CONSUMPTION']):
    """
    Construct input-output sequences for time series forecasting - multivariate output.

    Parameters:
    - df: pd.DataFrame, input time series data with datetime index
    - history_len: int, number of past time steps to use as input
    - horizon: int, number of future time steps to predict
    - step_size: int, step size between sequences
    - target_cols: list of str, columns to predict

    Returns:
    - X: np.ndarray, input sequences of shape (num_samples, num_features, history_len)
    - Y: np.ndarray, output sequences of shape (num_samples, len(target_cols), horizon)
    """
    data = df.values
    num_samples = (len(df) - history_len - horizon) // step_size + 1
    num_features = data.shape[1]
    num_targets = len(target_cols)

    X = np.zeros((num_samples, num_features, history_len))
    Y = np.zeros((num_samples, num_targets, horizon))

    for i in range(num_samples):
        start_x = i * step_size
        end_x = start_x + history_len
        start_y = end_x
        end_y = start_y + horizon

        X[i] = data[start_x:end_x].T  # Transpose to get shape (num_features, history_len)
        for j, col in enumerate(target_cols):
            Y[i, j] = data[start_y:end_y, df.columns.get_loc(col)]

    return X, Y
    
# Helper: preview plot on the eval snapshots
def preview_total_plot(model, eval_loader, timestamps, truth_raw, N_eval, L, H, stride, agg_mode, load_mean, load_std, save_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), writer=None, step=None, target_channel: int = 0):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            # Ensure shape (B, H) for each batch. For multi-target outputs, select target_channel
            if out.ndim == 3:   # (B, C, H)
                # select requested target channel
                out = out[:, target_channel, :]
            elif out.ndim == 1: # (H,) -> (1, H)
                out = out[None, :]
            preds.append(out)
    all_preds = np.concatenate(preds, axis=0)  # shape (num_windows, H_out)
    num_windows, H_out = all_preds.shape
    use_H = min(H, H_out)
    # Map each prediction window to timeline indices
    t_starts = np.arange(0, N_eval - L - H + 1, stride)
    if len(t_starts) != num_windows:
        t_starts = np.arange(num_windows)
    bags = defaultdict(list)
    for w, t0 in enumerate(t_starts):
        for h_step in range(use_H):
            t_idx = t0 + L + h_step
            if t_idx >= N_eval: break
            bags[t_idx].append(all_preds[w, h_step])
    pred_indices = sorted(bags.keys())
    if not pred_indices:
        return None
    # Aggregate overlapping predictions
    if agg_mode == "mean":
        agg_pred_norm = np.array([np.mean(bags[i]) for i in pred_indices], dtype=np.float32)
    else:  # "first"
        agg_pred_norm = np.array([bags[i][0] for i in pred_indices], dtype=np.float32)
    # Denormalize predictions
    agg_pred = agg_pred_norm * (load_std if load_std != 0 else 1.0) + load_mean
    # Align with ground truth
    dt_aligned   = timestamps[pred_indices]
    true_aligned = truth_raw[pred_indices]
    # Metrics
    diff = agg_pred - true_aligned
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2)); rmse = float(np.sqrt(mse))
    # Plot preview
    fig = plt.figure(figsize=(12,5))
    plt.plot(dt_aligned, true_aligned, label="True", linestyle="--")
    plt.plot(dt_aligned, agg_pred, label="Predicted", marker='o', linewidth=1)
    plt.title(f"Preview @ epoch {step or ''} (H={H}, stride={stride}, agg={agg_mode})\nMAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
    plt.xlabel("Datetime"); plt.ylabel("Load"); plt.xticks(rotation=45)
    plt.legend(); plt.grid(True); plt.tight_layout()
    fname = f"preview_total_epoch{step:04d}.png" if step is not None else "preview_total_latest.png"
    plt.savefig(os.path.join(save_dir, fname))
    if writer and step is not None:
        writer.add_figure("preview_total/pred_vs_true", fig, global_step=step)
    plt.close(fig)
    return mae, rmse  # return metrics if needed

