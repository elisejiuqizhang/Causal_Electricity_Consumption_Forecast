# exp/train/train_forecast.py
import os, json, math, argparse, random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ------------------ Utils ------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def time_features(df: pd.DataFrame, time_col="time") -> pd.DataFrame:
    """Add cyclical datetime features. Assumes time is parseable to pandas datetime."""
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])
    dt = df[time_col].dt
    df["hour"] = dt.hour
    df["dow"]  = dt.dayofweek
    df["doy"]  = dt.dayofyear
    df["month"]= dt.month

    # cyclical encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    df["month_sin"]= np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"]= np.cos(2*np.pi*df["month"]/12.0)
    return df

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

def plot_forecast(ts_time, y_true, y_pred, title, outpath):
    plt.figure()
    plt.plot(ts_time, y_true, label="actual")
    plt.plot(ts_time, y_pred, label="pred")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

# ------------------ Data ------------------

@dataclass
class SeriesConfig:
    city: str
    path: str  # CSV: must include columns ['time','load', <weather...>]

class RollingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # X: (N, L, D), Y: (N, H)
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_windows(df: pd.DataFrame, input_cols: List[str], target_col: str,
                  history: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Returns X: (N, L, D), Y: (N, H), and end times aligned to Y end.
    """
    values = df[input_cols + [target_col]].values
    times = df["time"].values
    N = len(df)
    L = history; H = horizon
    X_list, Y_list, T_list = [], [], []
    for end in range(L, N - H + 1):
        X_list.append(values[end-L:end, :len(input_cols)])
        Y_list.append(values[end:end+H, -1])  # last col is target
        T_list.append(times[end+H-1])
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, L, len(input_cols)))
    Y = np.stack(Y_list, axis=0) if Y_list else np.zeros((0, H))
    T = [pd.Timestamp(t) for t in T_list]
    return X, Y, T

def split_by_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    N = len(df)
    i_tr = int(N * train_ratio)
    i_va = int(N * (train_ratio + val_ratio))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]

# ------------------ Models ------------------


def make_model(model_name: str, input_dim: int, horizon: int,
               model_class: Optional[str], model_module: Optional[str],
               patchtst_kwargs: dict):
    if model_name.lower() in ["gru", "lstm"]:
        if model_module is None:
            model_module = "utils.forecast.rnn"  # default path in your repo
        if model_class is None:
            model_class = "GRUForecaster" if model_name.lower() == "gru" else "LSTMForecaster"
        import importlib
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        # Most RNN variants take parameters like: input_dim, hidden_dim, num_layers, horizon, ...
        # We only pass what we know; rest can be set via --rnn-kwargs
        model = ModelClass(input_dim=input_dim, horizon=horizon, **patchtst_kwargs)
        return model
    elif model_name.lower() == "tcn":
        if model_module is None:
            model_module = "utils.forecast.TCN"  # default path in your repo
        if model_class is None:
            model_class = "TCNModel"  # adjust if your class is named differently
    elif model_name.lower() == "mamba":
        if model_module is None:
            model_module = "utils.forecast.mamba"  # default path in your repo
        if model_class is None:
            model_class = "UniVForecaster"  # adjust if your class is named differently
    elif model_name.lower() == "patchtst":
        if model_module is None:
            model_module = "utils.forecast.PatchTST"  # default path in your repo
        if model_class is None:
            model_class = "PatchTSTModel"  # adjust if your class is named differently

        import importlib
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        # Most PatchTST variants take parameters like: c_in, c_out, seq_len, pred_len, ...
        # We only pass what we know; rest can be set via --patchtst-kwargs
        # Include input_dim and horizon as c_in / pred_len if your implementation uses those names.
        model = ModelClass(c_in=input_dim, pred_len=horizon, **patchtst_kwargs)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ------------------ Training ------------------

def train_loop(model, loaders, device, epochs=20, lr=1e-3, ckpt_path=None):
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, len(loaders["train"].dataset))

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= max(1, len(loaders["val"].dataset))

        print(f"[Epoch {ep:03d}] train MSE={tr_loss:.6f}  val MSE={va_loss:.6f}")

        if va_loss < best_val and ckpt_path is not None:
            best_val = va_loss
            ensure_dir(os.path.dirname(ckpt_path))
            torch.save(model.state_dict(), ckpt_path)
    return best_val

def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            yhat = model(xb)  # (B,H)
            preds.append(yhat.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,))

# ------------------ Pipeline ------------------

def load_city_csv(path: str, time_col="time") -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"{path} must include a '{time_col}' column.")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def prepare_city_frame(cfg: SeriesConfig, features: List[str], target_col="load") -> pd.DataFrame:
    df = load_city_csv(cfg.path)
    # always add time features
    df = time_features(df, time_col="time")
    # ensure required columns exist
    missing = [c for c in [target_col] + features if c not in df.columns]
    if missing:
        raise ValueError(f"{cfg.city}: missing columns {missing} in {cfg.path}")
    # final frame with time, target, and features
    keep = ["time", target_col] + features + [
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
    ]
    return df[keep].copy()

def standardize_train_val_test(train_df, val_df, test_df, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df[cols] = scaler.fit_transform(train_df[cols].to_numpy())
    val_df[cols]   = scaler.transform(val_df[cols].to_numpy())
    test_df[cols]  = scaler.transform(test_df[cols].to_numpy())
    return train_df, val_df, test_df, scaler

def make_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=128):
    ds_tr = RollingWindowDataset(Xtr, Ytr)
    ds_va = RollingWindowDataset(Xva, Yva)
    ds_te = RollingWindowDataset(Xte, Yte)
    ld = {
        "train": DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False),
        "val":   DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False),
        "test":  DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False),
    }
    return ld

def run_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Parse cities CSV mapping: you provide pairs --city-csv "CityA=path1.csv" --city-csv "CityB=path2.csv"
    series_cfgs = []
    for kv in args.city_csv:
        if "=" not in kv:
            raise ValueError(f"--city-csv must be City=path.csv, got: {kv}")
        city, path = kv.split("=", 1)
        series_cfgs.append(SeriesConfig(city=city.strip(), path=path.strip()))

    features = [f.strip() for f in args.features.split(",")] if args.features else []
    # datetime & historical load are always included (datetime is added inside, historical load is target_lag from windows)

    exp_dir = ensure_dir(os.path.join(args.save_dir, args.exp_name))
    ensure_dir(os.path.join(exp_dir, "figures"))
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best.pt")

    # Prepare data per city
    per_city_frames = {cfg.city: prepare_city_frame(cfg, features, target_col=args.target_col) for cfg in series_cfgs}

    # For total mode we’ll create a synthetic "TOTAL" series by summing target across cities
    if args.mode == "total":
        # align on time
        all_times = None
        for df in per_city_frames.values():
            all_times = df["time"] if all_times is None else all_times.to_frame().merge(df[["time"]], on="time", how="inner")["time"]
        frames_aligned = []
        for city, df in per_city_frames.items():
            frames_aligned.append(df[df["time"].isin(all_times)].sort_values("time"))
        # Sum targets; features: (a) you can sum, average, or choose one city’s weather; here we average features across cities.
        total_df = frames_aligned[0][["time"]].copy()
        total_df[args.target_col] = 0.0
        feat_cols = [c for c in frames_aligned[0].columns if c not in ["time", args.target_col]]
        for c in feat_cols: total_df[c] = 0.0
        for df in frames_aligned:
            total_df[args.target_col] += df[args.target_col].values
            for c in feat_cols:
                total_df[c] += df[c].values
        total_df[feat_cols] /= len(frames_aligned)
        per_city_frames = {"TOTAL": total_df}

    # Split each city into train/val/test by time, then stack if multi_city mode
    input_cols = [c for c in per_city_frames[next(iter(per_city_frames))].columns if c not in ["time", args.target_col]]
    city_splits = {}
    for city, df in per_city_frames.items():
        tr, va, te = split_by_time(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        tr, va, te, scaler = standardize_train_val_test(tr, va, te, input_cols)
        city_splits[city] = (tr, va, te, scaler)

    # Build windows per city
    city_windows = {}
    for city, (tr, va, te, scaler) in city_splits.items():
        Xtr, Ytr, Ttr = build_windows(tr, input_cols, args.target_col, args.history, args.horizon)
        Xva, Yva, Tva = build_windows(va, input_cols, args.target_col, args.history, args.horizon)
        Xte, Yte, Tte = build_windows(te, input_cols, args.target_col, args.history, args.horizon)
        city_windows[city] = dict(Xtr=Xtr, Ytr=Ytr, Ttr=Ttr, Xva=Xva, Yva=Yva, Tva=Tva, Xte=Xte, Yte=Yte, Tte=Tte)

    # Assemble loaders based on mode
    if args.mode == "per_city":
        if len(series_cfgs) != 1:
            raise ValueError("per_city mode expects exactly one --city-csv City=path.csv")
        city = series_cfgs[0].city
        W = city_windows[city]
        loaders = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)
        input_dim = W["Xtr"].shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.patchtst_kwargs).to(device)
        best_val = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)

        # Load best and predict test
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        yhat = predict(model, loaders["test"], device)  # (N,H)
        ytrue = city_windows[city]["Yte"]               # (N,H)

        # Evaluate on last step of horizon (or average)
        yhat_last = yhat[:, -1]; ytrue_last = ytrue[:, -1]
        metrics = compute_metrics(ytrue_last, yhat_last)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump({city: metrics}, f, indent=2)

        # Save CSV with timestamps and (optionally) full horizon flattened
        df_out = pd.DataFrame({
            "time": city_windows[city]["Tte"],
            "city": city,
            "y_true_last": ytrue_last,
            "y_pred_last": yhat_last
        })
        df_out.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)

        # Plot
        plot_forecast(df_out["time"], df_out["y_true_last"], df_out["y_pred_last"], f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

    elif args.mode == "multi_city":
        # concatenate all cities
        Xtr = np.concatenate([city_windows[c]["Xtr"] for c in city_windows], axis=0)
        Ytr = np.concatenate([city_windows[c]["Ytr"] for c in city_windows], axis=0)
        Xva = np.concatenate([city_windows[c]["Xva"] for c in city_windows], axis=0)
        Yva = np.concatenate([city_windows[c]["Yva"] for c in city_windows], axis=0)
        Xte = np.concatenate([city_windows[c]["Xte"] for c in city_windows], axis=0)
        Yte = np.concatenate([city_windows[c]["Yte"] for c in city_windows], axis=0)

        loaders = make_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, args.batch_size)
        input_dim = Xtr.shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.patchtst_kwargs).to(device)
        best_val = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Per-city evaluation (predict per-city to get per-city plots/CSVs)
        all_rows = []
        per_city_metrics = {}
        for city in city_windows:
            W = city_windows[city]
            test_loader = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)["test"]
            yhat = predict(model, test_loader, device)
            ytrue = W["Yte"]
            yhat_last, ytrue_last = yhat[:, -1], ytrue[:, -1]
            met = compute_metrics(ytrue_last, yhat_last)
            per_city_metrics[city] = met
            rows = pd.DataFrame({
                "time": W["Tte"],
                "city": city,
                "y_true_last": ytrue_last,
                "y_pred_last": yhat_last
            })
            all_rows.append(rows)
            plot_forecast(W["Tte"], ytrue_last, yhat_last, f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

        # Aggregate metrics over all cities
        big = pd.concat(all_rows, axis=0, ignore_index=True)
        agg = compute_metrics(big["y_true_last"].to_numpy(), big["y_pred_last"].to_numpy())
        per_city_metrics["_aggregate_"] = agg
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(per_city_metrics, f, indent=2)
        big.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)

    elif args.mode == "total":
        # There is only one series now: "TOTAL"
        city = "TOTAL"
        W = city_windows[city]
        loaders = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)
        input_dim = W["Xtr"].shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.patchtst_kwargs).to(device)
        best_val = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        yhat = predict(model, loaders["test"], device)
        ytrue = W["Yte"]
        yhat_last, ytrue_last = yhat[:, -1], ytrue[:, -1]
        metrics = compute_metrics(ytrue_last, yhat_last)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump({city: metrics}, f, indent=2)
        df_out = pd.DataFrame({
            "time": W["Tte"],
            "city": city,
            "y_true_last": ytrue_last,
            "y_pred_last": yhat_last
        })
        df_out.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)
        plot_forecast(W["Tte"], ytrue_last, yhat_last, f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save config for reproducibility
    cfg = {k: (v if k != "patchtst_kwargs" else v) for k, v in vars(args).items()}
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def parse_kwargs(kvs: List[str]) -> dict:
    out = {}
    for kv in kvs:
        if "=" not in kv: raise ValueError(f"Bad kwarg '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        # try to parse numeric/bool
        if v.lower() in ["true","false"]:
            v = v.lower() == "true"
        else:
            try:
                if "." in v: v = float(v)
                else: v = int(v)
            except:
                pass
        out[k] = v
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ontario Load Forecasting (per_city / multi_city / total)")
    p.add_argument("--mode", choices=["per_city","multi_city","total"], required=True)
    p.add_argument("--city-csv", action="append", required=True,
                   help="Repeatable: City=path/to/merged_city.csv (must include time, load, chosen weather columns)")
    p.add_argument("--features", type=str, default="",
                   help="Comma-separated weather feature names to include. (Datetime features are auto; load is target.)")
    p.add_argument("--target-col", type=str, default="load")
    p.add_argument("--history", type=int, default=168, help="lookback length (hours)")
    p.add_argument("--horizon", type=int, default=24, help="forecast horizon (hours)")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--model", choices=["gru","patchtst"], default="gru",
                   help="Use 'gru' to validate pipeline; switch to 'patchtst' when your module is verified.")
    p.add_argument("--model-module", type=str, default=None,
                   help="Python import path for your model module, e.g. utils.forecast.PatchTST")
    p.add_argument("--model-class", type=str, default=None,
                   help="Class name in the module, e.g. PatchTST")
    p.add_argument("--patchtst-kwarg", action="append", default=[],
                   help="Repeatable: pass key=value args to PatchTST (e.g., d_model=128, embed='timeF')")
    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--exp-name", type=str, default="debug")

    args = p.parse_args()
    args.patchtst_kwargs = parse_kwargs(args.patchtst_kwarg)
    run_mode(args)
