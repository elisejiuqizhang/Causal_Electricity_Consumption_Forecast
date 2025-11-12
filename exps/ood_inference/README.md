# OOD Inference Scripts

This directory contains scripts for testing pretrained forecasting models on Out-Of-Distribution (OOD) extreme weather windows.

## Overview

The OOD inference pipeline:
1. Loads pretrained models (GRU, TCN, or PatchTST)
2. Tests them on identified extreme weather windows
3. Compares performance on OOD conditions vs. normal conditions
4. Generates detailed metrics and predictions

## Files

- **`gru_ood_inference.py`**: GRU model inference script
- **`tcn_ood_inference.py`**: TCN model inference script  
- **`patchtst_ood_inference.py`**: PatchTST model inference script
- **`run_ood_inference.sh`**: Batch script to run all inferences
- **`README.md`**: This file

## Prerequisites

1. **Trained models** must exist in `outputs/forecast/per_region/{model_name}/`
2. **OOD windows** must be identified using `identify_ood_weather.py`
   - Files: `outputs/ood_analysis/ood_windows_{Region}_val.csv`

## Quick Start

### Run All Models (Toronto & Ottawa)

```bash
cd exps/ood_inference
chmod +x run_ood_inference.sh
./run_ood_inference.sh
```

This will test all three models (GRU, TCN, PatchTST) on both Toronto and Ottawa OOD windows.

### Run Individual Model

#### GRU Example
```bash
python gru_ood_inference.py \
    --region Toronto \
    --feature_set F2 \
    --fold 0 \
    --seed 97 \
    --ood_file ../../outputs/ood_analysis/ood_windows_Toronto_val.csv
```

#### TCN Example
```bash
python tcn_ood_inference.py \
    --region Ottawa \
    --feature_set F2 \
    --fold 0 \
    --seed 97 \
    --ood_file ../../outputs/ood_analysis/ood_windows_Ottawa_val.csv
```

#### PatchTST Example
```bash
python patchtst_ood_inference.py \
    --region Toronto \
    --feature_set F2 \
    --fold 0 \
    --seed 597 \
    --ood_file ../../outputs/ood_analysis/ood_windows_Toronto_val.csv
```

## Configuration

### Key Parameters

All scripts share these common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--region` | *Required* | Region name (Toronto, Ottawa, etc.) |
| `--feature_set` | F2 | Feature set (F0, F1, F2, F3) |
| `--fold` | 0 | Fold number to use |
| `--seed` | Model-specific | Random seed used during training |
| `--input_length` | 168 | Input sequence length (hours) |
| `--horizon` | 24 | Prediction horizon (hours) |
| `--stride` | 1 | Stride for predictions |
| `--batch_size` | 64 | Batch size for inference |
| `--ood_file` | *Required* | Path to OOD windows CSV |

### Model-Specific Parameters

**GRU:**
- `--d_model`: Hidden dimension (default: 64)
- `--n_layers`: Number of GRU layers (default: 4)
- `--dropout`: Dropout rate (default: 0.1)

**TCN:**
- `--hidden_channels`: Hidden channels (default: 64)
- `--levels`: Number of TCN levels (default: 4)
- `--kernel_size`: Kernel size (default: 3)
- `--dilation_base`: Dilation base (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

**PatchTST:**
- `--d_model`: Transformer dimension (default: 32)
- `--n_heads`: Attention heads (default: 4)
- `--n_layers`: Transformer layers (default: 3)
- `--patch_len`: Patch length (default: 16)
- `--patch_stride`: Patch stride (default: 8)
- `--dropout`: Dropout rate (default: 0.1)

### Training Configuration

The `--training_config` parameter must match the folder structure of your trained models:

```
outputs/forecast/per_region/{model_name}/{region}/
└── bs64_ep500_lr0.0001_tr0.93_vr0.07_pat20_esep0.0001/
    └── F2/
        └── {seed}/
            └── fold_0/
                ├── best_model.pth
                └── scaler_all.pkl
```

Default: `bs64_ep500_lr0.0001_tr0.93_vr0.07_pat20_esep0.0001`

## Output Structure

Results are saved in `outputs/ood_inference/{model_name}/`:

```
outputs/ood_inference/
├── gru/
│   ├── Toronto_F2_fold0_ood_metrics.csv
│   ├── Toronto_F2_fold0_ood_predictions.csv
│   ├── Toronto_F2_fold0_summary.txt
│   ├── Ottawa_F2_fold0_ood_metrics.csv
│   ├── Ottawa_F2_fold0_ood_predictions.csv
│   └── Ottawa_F2_fold0_summary.txt
├── tcn/
│   └── ...
└── patchtst/
    └── ...
```

### Output Files

1. **`*_ood_metrics.csv`**: Per-window summary
   - Columns: `window_idx`, `start_datetime`, `end_datetime`, `ood_fraction`, `MAE`, `RMSE`, `MAPE`, `SMAPE`, `n_predictions`

2. **`*_ood_predictions.csv`**: Timestep-level predictions
   - Columns: `window_idx`, `datetime`, `predicted_load`, `true_load`, `error`, `abs_error`

3. **`*_summary.txt`**: Text summary with:
   - Configuration details
   - Average metrics across all OOD windows
   - Per-window results table

## Analysis Workflow

### 1. Compare OOD vs. Normal Performance

```python
import pandas as pd

# Load OOD results
ood_metrics = pd.read_csv('outputs/ood_inference/gru/Toronto_F2_fold0_ood_metrics.csv')

# Load normal test results
normal_results = pd.read_csv('outputs/forecast/per_region/gru_single_train/Toronto/.../fold_0/test_predictions.csv')

# Compare
print(f"OOD MAE: {ood_metrics['MAE'].mean():.2f}")
print(f"Normal MAE: {calculate_mae(normal_results):.2f}")
```

### 2. Identify Challenging Windows

```python
# Sort by error
ood_metrics_sorted = ood_metrics.sort_values('MAPE', ascending=False)
print("Top 5 most challenging OOD windows:")
print(ood_metrics_sorted[['start_datetime', 'MAPE', 'ood_fraction']].head())
```

### 3. Analyze Error Patterns

```python
# Load detailed predictions
predictions = pd.read_csv('outputs/ood_inference/gru/Toronto_F2_fold0_ood_predictions.csv')
predictions['datetime'] = pd.to_datetime(predictions['datetime'])

# Group by hour of day
predictions['hour'] = predictions['datetime'].dt.hour
hourly_error = predictions.groupby('hour')['abs_error'].mean()

# Plot
import matplotlib.pyplot as plt
hourly_error.plot(kind='bar', title='Error by Hour of Day (OOD Windows)')
plt.ylabel('Mean Absolute Error')
plt.show()
```

### 4. Cross-Model Comparison

```python
# Load results from all models
gru_metrics = pd.read_csv('outputs/ood_inference/gru/Toronto_F2_fold0_ood_metrics.csv')
tcn_metrics = pd.read_csv('outputs/ood_inference/tcn/Toronto_F2_fold0_ood_metrics.csv')
patchtst_metrics = pd.read_csv('outputs/ood_inference/patchtst/Toronto_F2_fold0_ood_metrics.csv')

# Compare
comparison = pd.DataFrame({
    'GRU': [gru_metrics['MAE'].mean(), gru_metrics['RMSE'].mean(), gru_metrics['MAPE'].mean()],
    'TCN': [tcn_metrics['MAE'].mean(), tcn_metrics['RMSE'].mean(), tcn_metrics['MAPE'].mean()],
    'PatchTST': [patchtst_metrics['MAE'].mean(), patchtst_metrics['RMSE'].mean(), patchtst_metrics['MAPE'].mean()]
}, index=['MAE', 'RMSE', 'MAPE'])

print(comparison)
```

## Troubleshooting

### Model Not Found
- Verify models exist in `outputs/forecast/per_region/`
- Check that `--seed`, `--feature_set`, and `--fold` match training configuration
- Ensure `--training_config` matches folder name

### OOD Windows Not Found
- Run `identify_ood_weather.py` first to generate OOD windows
- Verify file exists: `outputs/ood_analysis/ood_windows_{Region}_val.csv`

### Hyperparameter Mismatch
- Model loading requires exact hyperparameters from training
- Check training logs or `overall_results.txt` for configuration
- Update parameters in script or `run_ood_inference.sh`

### Memory Issues
- Reduce `--batch_size` (e.g., to 32 or 16)
- Process one region/model at a time
- Use CPU if GPU memory insufficient

## Advanced Usage

### Test on Training OOD Windows

```bash
python gru_ood_inference.py \
    --region Toronto \
    --ood_file ../../outputs/ood_analysis/ood_windows_Toronto_train.csv \
    --output_dir ../../outputs/ood_inference_train/gru
```

### Test Different Folds

```bash
# Test all 3 folds
for fold in 0 1 2; do
    python gru_ood_inference.py \
        --region Toronto \
        --fold $fold \
        --ood_file ../../outputs/ood_analysis/ood_windows_Toronto_val.csv
done
```

### Custom Feature Sets

```bash
# Test model trained on F1 (all features)
python tcn_ood_inference.py \
    --region Toronto \
    --feature_set F1 \
    --ood_file ../../outputs/ood_analysis/ood_windows_Toronto_val.csv
```

## Citation & References

This inference pipeline is designed to evaluate model robustness on extreme weather conditions identified using the 5th/95th percentile method with 24-hour sliding windows.

For methodology details, see:
- `outputs/ood_analysis/OOD_SUMMARY_Toronto_Ottawa.md`
- `exps/ood_weather/identify_ood_weather.py`

## Support

For issues or questions:
1. Check model paths and configurations
2. Verify OOD windows exist and are valid
3. Ensure hyperparameters match training
4. Review output logs for specific error messages
