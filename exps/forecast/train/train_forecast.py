# exp/train/train_forecast.py
import os, sys, json, math, argparse, random
ROOT= os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.data_utils.datetime_utils import time_features
from utils.data_utils.processing import TargetScaler, split_by_time, build_windows
from utils.forecast.metrics import compute_metrics, per_horizon_metrics

# ------------------ Utils ------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def make_input_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    raise ValueError(f"Unknown scaler kind: {kind}")

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

    # avoid NaN
    df = df[1:51739]  # to avoid any potential issues with NaN

    return df

def prepare_city_frame(cfg: SeriesConfig, features: List[str], target_col="load", include_past_load=True) -> pd.DataFrame:
    df = merge_city_elec_meteo(cfg.elec_path, cfg.meteo_path)
    df = time_features(df, time_col="time")

    # validate availability
    missing = [c for c in [target_col] + features if c not in df.columns]
    if missing:
        raise ValueError(f"{cfg.city}: missing columns {missing} in merged CSVs ({cfg.elec_path}, {cfg.meteo_path})")

    keep = ["time", target_col] + features + [
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos", "is_holiday"
    ]
    if include_past_load:
        df["load_past"] = df[target_col]
        keep.append("load_past")
    return df[keep].copy()

def standardize_train_val_test(train_df, val_df, test_df, cols, scaler=None, scaler_kind="standard"):
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    if scaler_kind == "none":
        # no scaling requested
        return train_df, val_df, test_df, None
    if scaler is None:
        scaler = make_input_scaler(scaler_kind)
        scaler = scaler.fit(train_df[cols].to_numpy())
    train_df.loc[:, cols] = scaler.transform(train_df[cols].to_numpy())
    val_df.loc[:, cols]   = scaler.transform(val_df[cols].to_numpy())
    test_df.loc[:, cols]  = scaler.transform(test_df[cols].to_numpy())
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
        if model_class is None:  model_class = "MambaForecaster"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        return ModelClass(input_dim=input_dim, horizon=horizon, **model_kwargs)

    elif name == "patchtst":
        if model_module is None: model_module = "utils.forecast.PatchTST"
        if model_class is None:  model_class = "PatchTSTModel"
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
        return ModelClass(input_dim=input_dim, output_horizon=horizon, **model_kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ------------------ Training ------------------

def train_loop(model, loaders, device, epochs=20, lr=1e-3, ckpt_path=None, patience=5, tol=1e-5):
    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float("inf")
    bad=0
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

        # print(f"[Epoch {ep:03d}] train MSE={tr_loss:.6f}  val MSE={va_loss:.6f}")
        print(f"[Epoch {ep:03d}] train MSE={tr_loss:.3f} (RMSE={tr_loss**0.5:.3f})  "
                f"val MSE={va_loss:.3f} (RMSE={va_loss**0.5:.3f})")


        if va_loss < best_val-tol and ckpt_path is not None:
            best_val = va_loss
            ensure_dir(os.path.dirname(ckpt_path))
            torch.save(model.state_dict(), ckpt_path)
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stopping at epoch {ep} (best val MSE={best_val:.6f}, best val RMSE={best_val**0.5:.3f})")
                break
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
    # for city, df in per_city_frames.items():
    #     tr, va, te = split_by_time(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    #     tr, va, te, scaler = standardize_train_val_test(tr, va, te, input_cols, scaler_kind=args.scaler_kind)
    #     city_splits[city] = (tr, va, te, scaler)
    for city, df in per_city_frames.items():
        tr, va, te = split_by_time(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

        # Input scaling (per-series)
        tr, va, te, in_scaler = standardize_train_val_test(
            tr, va, te, input_cols, scaler=None, scaler_kind=args.input_scaler
        )

        # Target scaling (optional, fit on TRAIN targets only)
        tgt_scaler = None
        if args.scale_target:
            y_tr = tr[args.target_col].to_numpy()
            tgt_scaler = TargetScaler(mean=float(y_tr.mean()), std=float(y_tr.std()))
            # not write back scaled targets into df; scale Y arrays after windowing

        city_splits[city] = (tr, va, te, in_scaler, tgt_scaler)


    # Build windows per city
    city_windows = {}
    for city, (tr, va, te, in_scaler, tgt_scaler) in city_splits.items():
        Xtr, Ytr, Ttr = build_windows(tr, input_cols, args.target_col, args.history, args.horizon)
        Xva, Yva, Tva = build_windows(va, input_cols, args.target_col, args.history, args.horizon)
        Xte, Yte, Tte = build_windows(te, input_cols, args.target_col, args.history, args.horizon)

        if args.scale_target and tgt_scaler is not None:
            Ytr_s = tgt_scaler.transform(Ytr); Yva_s = tgt_scaler.transform(Yva); Yte_s = tgt_scaler.transform(Yte)
        else:
            Ytr_s, Yva_s, Yte_s = Ytr, Yva, Yte

        city_windows[city] = dict(
            Xtr=Xtr, Ytr=Ytr_s, Ttr=Ttr,
            Xva=Xva, Yva=Yva_s, Tva=Tva,
            Xte=Xte, Yte=Yte_s, Tte=Tte,
            Yraw_tr=Ytr, Yraw_va=Yva, Yraw_te=Yte,  # keep raw for metrics/CSV when needed
            tgt_scaler=tgt_scaler
        )


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

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)

        yhat = predict(model, loaders["test"], device)   # scaled-space or raw depending on args
        W = city_windows[city]
        if args.scale_target and W["tgt_scaler"] is not None:
            yhat_raw = W["tgt_scaler"].inverse(yhat)
            ytrue_raw = W["Yraw_te"]
        else:
            yhat_raw = yhat
            ytrue_raw = W["Yte"]
        # metrics on raw units:
        metrics_H = per_horizon_metrics(ytrue_raw, yhat_raw)  # CSVs use yhat_raw / ytrue_raw

        y_hat = yhat_raw
        ytrue = ytrue_raw

        with open(os.path.join(exp_dir, "metrics_per_horizon.json"), "w") as f:
            json.dump({city: metrics_H}, f, indent=2)
            # (N,H) -> long form for CSV
        dfH = (pd.DataFrame(yhat, columns=[f"pred_h{h+1}" for h in range(yhat.shape[1])])
            .assign(time=W["Tte"], city=city))
        dfH_true = (pd.DataFrame(ytrue, columns=[f"true_h{h+1}" for h in range(ytrue.shape[1])]))
        dfH = pd.concat([dfH[["time","city"]], dfH_true, dfH.filter(like="pred_")], axis=1)
        dfH.to_csv(os.path.join(exp_dir, "preds_test_full_horizon.csv"), index=False)

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
        input_cols = [c for c in per_city_frames[next(iter(per_city_frames))].columns if c not in ["time", args.target_col]]

        # 1) split (no scaling yet)
        split_map = {}
        for city, df in per_city_frames.items():
            tr, va, te = split_by_time(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
            split_map[city] = (tr.copy(), va.copy(), te.copy())

        # 2) build the scaler
        global_in_scaler = None
        if args.global_scaler and args.input_scaler != "none":
            input_scaler = make_input_scaler(args.input_scaler)
            concat_train = pd.concat([split_map[c][0][input_cols] for c in split_map], axis=0)
            global_in_scaler = input_scaler.fit(concat_train.to_numpy())

        # 3) apply scaling (global or per-series)
        city_splits = {}
        for city, (tr, va, te) in split_map.items():
            if global_in_scaler is not None:
                tr_s, va_s, te_s, in_scaler = standardize_train_val_test(
                    tr, va, te, input_cols, scaler=global_in_scaler, scaler_kind=args.input_scaler
                )
            else:
                tr_s, va_s, te_s, in_scaler = standardize_train_val_test(
                    tr, va, te, input_cols, scaler=None, scaler_kind=args.input_scaler
                )

            # Target scaling per city (optional)
            tgt_scaler = None
            if args.scale_target:
                y_tr = tr_s[args.target_col].to_numpy()  # NOTE: target was not scaled; safe
                tgt_scaler = TargetScaler(mean=float(y_tr.mean()), std=float(y_tr.std()))

            city_splits[city] = (tr_s, va_s, te_s, in_scaler, tgt_scaler)



        Xtr = np.concatenate([city_windows[c]["Xtr"] for c in city_windows], axis=0)
        Ytr = np.concatenate([city_windows[c]["Ytr"] for c in city_windows], axis=0)
        Xva = np.concatenate([city_windows[c]["Xva"] for c in city_windows], axis=0)
        Yva = np.concatenate([city_windows[c]["Yva"] for c in city_windows], axis=0)
        Xte = np.concatenate([city_windows[c]["Xte"] for c in city_windows], axis=0)
        Yte = np.concatenate([city_windows[c]["Yte"] for c in city_windows], axis=0)


        model = make_model(args.model, input_dim, args.horizon, args.model_class, args.model_module, args.model_kwargs).to(device)
        _ = train_loop(model, loaders, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)


        all_rows = []
        per_city_metrics = {}
        for city in city_windows:
            W = city_windows[city]
            test_loader = make_loaders(W["Xtr"], W["Ytr"], W["Xva"], W["Yva"], W["Xte"], W["Yte"], args.batch_size)["test"]
            yhat = predict(model, test_loader, device)
            ytrue = W["Yte"]

            metrics_H = per_horizon_metrics(ytrue, yhat)
            with open(os.path.join(exp_dir, "metrics_per_horizon.json"), "w") as f:
                json.dump({city: metrics_H}, f, indent=2)
            # (N,H) -> long form for CSV
            dfH = (pd.DataFrame(yhat, columns=[f"pred_h{h+1}" for h in range(yhat.shape[1])])
                .assign(time=W["Tte"], city=city))
            dfH_true = (pd.DataFrame(ytrue, columns=[f"true_h{h+1}" for h in range(ytrue.shape[1])]))
            dfH = pd.concat([dfH[["time","city"]], dfH_true, dfH.filter(like="pred_")], axis=1)
            dfH.to_csv(os.path.join(exp_dir, "preds_test_full_horizon.csv"), index=False)


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
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)

        yhat = predict(model, loaders["test"], device)   # scaled-space or raw depending on args
        W = city_windows[city]
        if args.scale_target and W["tgt_scaler"] is not None:
            yhat_raw = W["tgt_scaler"].inverse(yhat)
            ytrue_raw = W["Yraw_te"]
        else:
            yhat_raw = yhat
            ytrue_raw = W["Yte"]

        # metrics on raw units:
        metrics_H = per_horizon_metrics(ytrue_raw, yhat_raw)
        # CSVs use yhat_raw / ytrue_raw

        y_hat = yhat_raw
        ytrue = ytrue_raw

        with open(os.path.join(exp_dir, "metrics_per_horizon.json"), "w") as f:
            json.dump({city: metrics_H}, f, indent=2)
        # (N,H) -> long form for CSV
        dfH = (pd.DataFrame(yhat, columns=[f"pred_h{h+1}" for h in range(yhat.shape[1])])
            .assign(time=W["Tte"], city=city))
        dfH_true = (pd.DataFrame(ytrue, columns=[f"true_h{h+1}" for h in range(ytrue.shape[1])]))
        dfH = pd.concat([dfH[["time","city"]], dfH_true, dfH.filter(like="pred_")], axis=1)
        dfH.to_csv(os.path.join(exp_dir, "preds_test_full_horizon.csv"), index=False)

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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--input-scaler", choices=["standard","minmax","none"], default="standard",
               help="How to scale input features (fitted on TRAIN, applied to VAL/TEST).")
    p.add_argument("--scale-target", action="store_true",
                help="If set, scale target on TRAIN and invert predictions before metrics/CSV.")
    p.add_argument("--global-scaler", action="store_true",
                help="If set in multi_city, fit ONE scaler on the concatenated TRAIN slices of all cities.")
    p.add_argument("--save-scalers", action="store_true",
                help="Persist fitted scalers (and target stats) to exp_dir for reproducibility.")


    p.add_argument("--model", choices=["gru","lstm","tcn","mamba","patchtst"], default="gru",
                   help="Backbone to use.")
    p.add_argument("--model-module", type=str, default=None,
                   help="Python import path for your model module, e.g. utils.forecast.PatchTST")
    p.add_argument("--model-class", type=str, default=None,
                   help="Class name in the module, e.g. PatchTSTModel")
    p.add_argument("--model-kwarg", action="append", default=[],
                   help="Repeatable: pass key=value args to the model (e.g., num_layers=2 hidden_size=256)")

    p.add_argument("--save-dir", type=str, default="outputs/forecast/train_runs")
    p.add_argument("--exp-name", type=str, default="debug")

    args = p.parse_args()
    args.model_kwargs = parse_kwargs(args.model_kwarg)
    run_mode(args)
