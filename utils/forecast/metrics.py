import numpy as np
import math
from typing import Dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(math.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
    smape = float(np.mean(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)
    ybar = float(np.mean(y_true))
    r2 = float(1.0 - (np.sum((y_true - y_pred)**2) / (np.sum((y_true - ybar)**2) + eps)))
    peak = float(np.max(y_true) + eps)
    mae_pct_peak = float(mae / peak * 100.0)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, sMAPE=smape, R2=r2, MAE_pct_peak=mae_pct_peak)

def per_horizon_metrics(y_true, y_pred):
    # y_* shape: (N, H)
    table = []
    for h in range(y_true.shape[1]):
        m = compute_metrics(y_true[:, h], y_pred[:, h])
        m["horizon"] = h+1
        table.append(m)
    return table
