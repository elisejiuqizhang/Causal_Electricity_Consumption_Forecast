import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class TargetScaler:
    mean: float
    std: float
    kind: str = "standard"  # future-proof (could extend to minmax)

    def transform(self, y):  # y: (N,H)
        return (y - self.mean) / (self.std + 1e-8)

    def inverse(self, yhat):
        return yhat * (self.std + 1e-8) + self.mean

# # min max scaling
# def min_max_scale(series):
#     """
#     returns scaled series, min, max
#     """
#     min_val = series.min()
#     max_val = series.max()
#     scaled_series = (series - min_val) / (max_val - min_val)
#     return scaled_series, min_val, max_val

def split_by_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    N = len(df)
    i_tr = int(N * train_ratio)
    i_va = int(N * (train_ratio + val_ratio))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]

def build_windows(df: pd.DataFrame, input_cols: List[str], target_col: str,
                  history: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Returns:
      X: (N, L, D) built from 'input_cols'
      Y: (N, H) from 'target_col'
      T: list of timestamps aligned to the end of each Y window
    """
    # ensure target_col appears exactly once in the projection
    all_cols = input_cols + ([target_col] if target_col not in input_cols else [])
    values = df[all_cols].values
    times = df["time"].values
    L, H = history, horizon
    D = len(input_cols)

    X_list, Y_list, T_list = [], [], []
    N = len(df)
    # y_col index in values is either D (if we appended) or the index within input_cols (dedup guard)
    y_col = all_cols.index(target_col)
    for end in range(L, N - H + 1):
        X_list.append(values[end-L:end, :D])                   # (L, D)
        Y_list.append(values[end:end+H, y_col])                # (H,)
        T_list.append(times[end+H-1])

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, L, D))
    Y = np.stack(Y_list, axis=0) if Y_list else np.zeros((0, H))
    T = [pd.Timestamp(t) for t in T_list]
    return X, Y, T



def time_delay_embed(ts, tau, emd, L=None):
    """ Time-delay embedding of a univariate time series.
    Args:
        ts   time series (univariate)
        tau  time delay
        emd  embedding dimension
        L    max length/time length, if None then full length
        
    Returns:
        An numpy array. Shape: [L-(emd-1)*tau, emd].

        

        Each row is one sample that represents a point (a certain time index t) on the embeddng;
        the elements of each sample are the values at time indices: 
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold
        
        Array dimensions: [number of samples, embedding dimension].
        Number of samples equals L-(emd-1)*tau.
    """

    if L is None:
        L = len(ts)
    resultArr=np.zeros((L-(emd-1)*tau,emd))
    for t in range((emd - 1) * tau, L):
        for i in range(emd):
            resultArr[t - (emd - 1) * tau][i] = ts[t - i * tau]
    return resultArr



def partial_corr(x, y, cond):
    """ Partial correlation between x and y conditionned on cond.
    ref: https://github.com/PengTao-HUST/crossmapy/blob/master/crossmapy/utils.py

    Parameters
    ----------
    x: 2D array [num_samples, dim]
        First variable.
    y: 2D array [num_samples, dim]
        Second variable.
    cond: 2D array [num_samples, dim]
        Conditioning variable.
    """  
    z = cond

    partial_corr = np.zeros(x.shape[1])

    for i in range(z.shape[1]):
        r_xy = np.corrcoef(x[:, i], y[:, i])[0, 1]
        r_xz = np.corrcoef(x[:, i], z[:, i])[0, 1]
        r_yz = np.corrcoef(y[:, i], z[:, i])[0, 1]

        partial_corr[i] = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    return partial_corr


def corr(x, y):
    """ Pearson's correlation between x and y.
     2D array [num_samples(timestamps), dim]
    """

    corr = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        corr[i] = np.corrcoef(x[:, i], y[:, i])[0, 1]

    return corr
