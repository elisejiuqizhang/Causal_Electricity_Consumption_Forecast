# exp/train/train_forecast.py
import os, sys, json, math, argparse, random
ROOT= os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)

from dataclasses import dataclass
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
    elec_path: str  # must include columns ['time', target_col]
    meteo_path: str # must include columns ['time', <weather...>]

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

def split_by_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    N = len(df)
    i_tr = int(N * train_ratio)
    i_va = int(N * (train_ratio + val_ratio))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]

def parse_city_kv_list(items: List[str]) -> Dict[str, str]:
    out = {}
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"Expect City=path, got: {kv}")
        city, path = kv.split("=", 1)
        out[city.strip()] = path.strip()
    return out

def load_csv(path: str, time_col="time") -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"{path} must include a '{time_col}' column.")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def merge_city_elec_meteo(elec_csv: str, meteo_csv: str) -> pd.DataFrame:
    de = load_csv(elec_csv, time_col='TIMESTAMP').rename(columns={'TIMESTAMP':'time'})
    dm = load_csv(meteo_csv, time_col='time')
    # inner join to ensure aligned hours
    df = de.merge(dm, on='time', how="inner")
    df = df.sort_values('time').reset_index(drop=True)
    return df

def prepare_city_frame(cfg: SeriesConfig, features: List[str], target_col="load", include_past_load=True) -> pd.DataFrame:
    df = merge_city_elec_meteo(cfg.elec_path, cfg.meteo_path)
    df = time_features(df, time_col="time")

    # validate availability
    missing = [c for c in [target_col] + features if c not in df.columns]
    if missing:
        raise ValueError(f"{cfg.city}: missing columns {missing} in merged CSVs ({cfg.elec_path}, {cfg.meteo_path})")

    keep = ["time", target_col] + features + [
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
    ]
    if include_past_load:
        df["load_past"] = df[target_col]
        keep.append("load_past")

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

# ------------------ Models ------------------

def make_model(model_name: str, input_dim: int, horizon: int,
               model_class: Optional[str], model_module: Optional[str],
               model_kwargs: dict):

    import importlib
    name = model_name.lower()

    if name in ["gru", "lstm"]:
        if model_module is None: model_module = "utils.forecast.rnn"
        if model_class is None:
            model_class = "GRUForecaster" if name == "gru" else "LSTMForecaster"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        return ModelClass(input_dim=input_dim, horizon=horizon, **model_kwargs)

    elif name == "tcn":
        if model_module is None: model_module = "utils.forecast.TCN"
        if model_class is None:  model_class = "TCNModel"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        return ModelClass(input_dim=input_dim, horizon=horizon, **model_kwargs)

    elif name == "mamba":
        if model_module is None: model_module = "utils.forecast.mamba"
        if model_class is None:  model_class = "UniVForecaster"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        return ModelClass(input_dim=input_dim, horizon=horizon, **model_kwargs)

    elif name == "patchtst":
        if model_module is None: model_module = "utils.forecast.PatchTST"
        if model_class is None:  model_class = "PatchTSTModel"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        # Many PatchTST impls expect c_in and pred_len (plus others in kwargs)
        return ModelClass(c_in=input_dim, pred_len=horizon, **model_kwargs)

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
            # squeeze if model returns (B,H,1)
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
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
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
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
            yhat = model(xb)  # (B,H) or (B,H,1)
            if yhat.ndim == 3 and yhat.shape[-1] == 1:
                yhat = yhat.squeeze(-1)
            preds.append(yhat.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,))

# ------------------ Pipeline ------------------

def run_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Parse city->paths (must match city sets)
    elec_map = parse_city_kv_list(args.city_electricity_csv)
    meteo_map = parse_city_kv_list(args.city_meteo_csv)
    if set(elec_map.keys()) != set(meteo_map.keys()):
        raise ValueError(f"City sets differ between electricity and meteo CSVs.\n"
                         f"Electricity: {sorted(elec_map.keys())}\nMeteo: {sorted(meteo_map.keys())}")
    series_cfgs = [SeriesConfig(city=c, elec_path=elec_map[c], meteo_path=meteo_map[c]) for c in sorted(elec_map.keys())]

    features = [f.strip() for f in args.features.split(",")] if args.features else []
    exp_dir = ensure_dir(os.path.join(args.save_dir, args.exp_name))
    ensure_dir(os.path.join(exp_dir, "figures"))
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best.pt")

    # Prepare merged frames per city
    per_city_frames = {
        cfg.city: prepare_city_frame(cfg, features, target_col=args.target_col, include_past_load=args.include_past_load)
        for cfg in series_cfgs
    }

    # Total mode: align times and aggregate
    if args.mode == "total":
        # Find common times across all cities
        common = None
        for df in per_city_frames.values():
            common = df[["time"]] if common is None else common.merge(df[["time"]], on="time", how="inner")
        common_times = common["time"]

        aligned = []
        for city, df in per_city_frames.items():
            aligned.append(df[df["time"].isin(common_times)].sort_values("time"))

        total_df = aligned[0][["time"]].copy()
        total_df[args.target_col] = 0.0
        feat_cols = [c for c in aligned[0].columns if c not in ["time", args.target_col]]
        for c in feat_cols: total_df[c] = 0.0
        for df in aligned:
            total_df[args.target_col] += df[args.target_col].to_numpy()
            for c in feat_cols:
                total_df[c] += df[c].to_numpy()
        total_df[feat_cols] /= len(aligned)
        per_city_frames = {"TOTAL": total_df}

    # Split each city by time; standardize only inputs
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
            raise ValueError("per_city mode expects exactly one city in the provided CSV args.")
        city = series_cfgs[0].city
        W = city_windows[city]
        loaders = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)
        input_dim = W["Xtr"].shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.model_kwargs).to(device)
        _ = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path): model.load_state_dict(torch.load(ckpt_path, map_location=device))
        yhat = predict(model, loaders["test"], device)  # (N,H)
        ytrue = W["Yte"]                                # (N,H)

        yhat_last = yhat[:, -1]; ytrue_last = ytrue[:, -1]
        metrics = compute_metrics(ytrue_last, yhat_last)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump({city: metrics}, f, indent=2)
        df_out = pd.DataFrame({
            "time": W["Tte"], "city": city,
            "y_true_last": ytrue_last, "y_pred_last": yhat_last
        })
        df_out.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)
        plot_forecast(W["Tte"], ytrue_last, yhat_last, f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

    elif args.mode == "multi_city":
        Xtr = np.concatenate([city_windows[c]["Xtr"] for c in city_windows], axis=0)
        Ytr = np.concatenate([city_windows[c]["Ytr"] for c in city_windows], axis=0)
        Xva = np.concatenate([city_windows[c]["Xva"] for c in city_windows], axis=0)
        Yva = np.concatenate([city_windows[c]["Yva"] for c in city_windows], axis=0)
        Xte = np.concatenate([city_windows[c]["Xte"] for c in city_windows], axis=0)
        Yte = np.concatenate([city_windows[c]["Yte"] for c in city_windows], axis=0)

        loaders = make_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, args.batch_size)
        input_dim = Xtr.shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.model_kwargs).to(device)
        _ = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path): model.load_state_dict(torch.load(ckpt_path, map_location=device))

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
            rows = pd.DataFrame({"time": W["Tte"], "city": city,
                                 "y_true_last": ytrue_last, "y_pred_last": yhat_last})
            all_rows.append(rows)
            plot_forecast(W["Tte"], ytrue_last, yhat_last, f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

        big = pd.concat(all_rows, axis=0, ignore_index=True)
        agg = compute_metrics(big["y_true_last"].to_numpy(), big["y_pred_last"].to_numpy())
        per_city_metrics["_aggregate_"] = agg
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(per_city_metrics, f, indent=2)
        big.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)

    elif args.mode == "total":
        city = "TOTAL"
        W = city_windows[city]
        loaders = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)
        input_dim = W["Xtr"].shape[-1]
        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.model_kwargs).to(device)
        _ = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path): model.load_state_dict(torch.load(ckpt_path, map_location=device))
        yhat = predict(model, loaders["test"], device)
        ytrue = W["Yte"]
        yhat_last, ytrue_last = yhat[:, -1], ytrue[:, -1]
        metrics = compute_metrics(ytrue_last, yhat_last)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump({city: metrics}, f, indent=2)
        df_out = pd.DataFrame({"time": W["Tte"], "city": city,
                               "y_true_last": ytrue_last, "y_pred_last": yhat_last})
        df_out.to_csv(os.path.join(exp_dir, "preds_test.csv"), index=False)
        plot_forecast(W["Tte"], ytrue_last, yhat_last, f"{city} - Test (last-step)", os.path.join(exp_dir, "figures", f"{city}_test_last.png"))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save config for reproducibility
    cfg = {k: (v if k != "model_kwargs" else v) for k, v in vars(args).items()}
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# ------------------ CLI ------------------

def parse_kwargs(kvs: List[str]) -> dict:
    out = {}
    for kv in kvs:
        if "=" not in kv: raise ValueError(f"Bad kwarg '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        if v.lower() in ["true","false"]:
            v = (v.lower() == "true")
        else:
            try:
                v = float(v) if "." in v else int(v)
            except:
                pass
        out[k] = v
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ontario Load Forecasting (per_city / multi_city / total)")

    # Separate CSVs for electricity and meteo
    p.add_argument("--city-electricity-csv", dest="city_electricity_csv", action="append", required=True,
                   help="Repeatable: City=path/to/elec.csv (must include time and target column)")
    p.add_argument("--city-meteo-csv", dest="city_meteo_csv", action="append", required=True,
                   help="Repeatable: City=path/to/meteo.csv (must include time and specified weather columns)")

    p.add_argument("--mode", choices=["per_city","multi_city","total"], required=True)
    p.add_argument("--features", type=str, default="",
                   help="Comma-separated weather feature names from the meteo CSVs. Datetime features are auto; load_past is optional.")
    p.add_argument("--target-col", type=str, default="TOTAL_CONSUMPTION")
    p.add_argument("--include-past-load", action="store_true", default=True,
                   help="Adds 'load_past' as an input feature (copy of target series). Target itself is never standardized.")
    p.add_argument("--history", type=int, default=168, help="lookback length (hours)")
    p.add_argument("--horizon", type=int, default=24, help="forecast horizon (hours)")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--model", choices=["gru","lstm","tcn","mamba","patchtst"], default="gru",
                   help="Backbone to use.")
    p.add_argument("--model-module", type=str, default=None,
                   help="Python import path for your model module, e.g. utils.forecast.PatchTST")
    p.add_argument("--model-class", type=str, default=None,
                   help="Class name in the module, e.g. PatchTSTModel")
    p.add_argument("--model-kwarg", action="append", default=[],
                   help="Repeatable: pass key=value args to the model (e.g., d_model=128 embed=timeF hidden_dim=256)")

    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--exp-name", type=str, default="debug")

    args = p.parse_args()
    args.model_kwargs = parse_kwargs(args.model_kwarg)
    run_mode(args)
