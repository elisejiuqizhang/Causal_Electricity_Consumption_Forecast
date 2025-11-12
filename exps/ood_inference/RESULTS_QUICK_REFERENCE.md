# OOD Inference - Quick Reference Card

**Last Updated**: November 2024  
**Status**: ✅ All 6 combinations completed (3 models × 2 regions)

---

## 📊 Results at a Glance

### Toronto (11 OOD windows)

| Model | OOD MAPE | vs Normal | Status |
|-------|----------|-----------|--------|
| **GRU** | 4.52% ± 1.63% | -22.8% ✅ | BETTER than normal |
| **TCN** | 4.81% ± 0.95% | +1.4% ⚠️ | Slightly worse |
| **PatchTST** | 10.34% ± 3.20% | N/A ❌ | Poor performance |

**Winner**: GRU (4.52% MAPE)

---

### Ottawa (13 OOD windows)

| Model | OOD MAPE | vs Normal | Status |
|-------|----------|-----------|--------|
| **GRU** | 15.46% ± 5.10% | +148.4% ❌ | Catastrophic failure |
| **TCN** | 7.95% ± 2.35% | +28.1% ⚠️ | Acceptable degradation |
| **PatchTST** | 5.28% ± 3.25% | N/A ✅ | Best performance |

**Winner**: PatchTST (5.28% MAPE)

---

## 🏆 Overall Winner: **TCN**

**Rationale**: 
- Consistent across both regions (4.81% Toronto, 7.95% Ottawa)
- Manageable degradation (+1.4% to +28%)
- Low variance = reliable predictions
- Production-ready

---

## 🚨 Key Findings

1. **Toronto Paradox**: Models perform BETTER on OOD than normal test
   - Suggests OOD definition may need refinement
   - Or test set has harder periods

2. **Ottawa Challenge**: Severe degradation for all models
   - Smaller dataset → overfitting
   - GRU fails worst (+175% MAE)

3. **Model-Region Mismatch**:
   - GRU: Great for Toronto, terrible for Ottawa
   - PatchTST: Terrible for Toronto, great for Ottawa
   - TCN: Good for both

---

## 📁 Output Files

### Per-Model Results
```
outputs/ood_inference/{model}/
├── Toronto_F2_fold0_ood_metrics.csv     # 11 windows summary
├── Toronto_F2_fold0_ood_predictions.csv # Timestep-level predictions
├── Toronto_F2_fold0_summary.txt         # Text report
├── Ottawa_F2_fold0_ood_metrics.csv      # 13 windows summary
├── Ottawa_F2_fold0_ood_predictions.csv  # Timestep-level predictions
└── Ottawa_F2_fold0_summary.txt          # Text report
```

### Comparison Analysis
```
outputs/ood_analysis/
├── ood_vs_normal_comparison_F2_fold0.csv        # Detailed comparison
├── ood_vs_normal_summary_F2_fold0.txt           # Summary statistics
└── ood_degradation_comparison_F2_fold0.png      # Visualization
```

---

## 🔧 Reproduce Results

```bash
# 1. Run all OOD inference (takes ~10 minutes)
./exps/ood_inference/run_ood_inference.sh

# 2. Generate comparison report
python exps/ood_inference/compare_ood_normal.py \
    --regions Toronto Ottawa \
    --models gru tcn patchtst \
    --feature_set F2 \
    --fold 0
```

---

## ⚙️ Model Configurations

| Param | GRU | TCN | PatchTST |
|-------|-----|-----|----------|
| **d_model** | 64 | 64 (hidden_channels) | 64 |
| **n_layers** | 4 | 4 (levels) | 3 |
| **n_heads** | - | - | 4 |
| **patch_len** | - | - | 16 |
| **patch_stride** | - | - | 8 |
| **kernel_size** | - | 3 | - |
| **seed** | 97 | 97 | 597 |

---

## 🌦️ OOD Windows Summary

### Toronto Extreme Events (11 windows)
- **Date Range**: Oct 2023 - Mar 2024
- **Primary Events**: Cold waves, heavy precipitation
- **Worst Window**: 2024-02-24 (GRU MAPE=7.80%)
- **Best Window**: 2024-01-26 (GRU MAPE=2.80%)

### Ottawa Extreme Events (13 windows)
- **Date Range**: Oct 2023 - Mar 2024
- **Primary Events**: Cold snaps, heavy snowfall, freezing rain
- **Worst Window**: 2024-02-19 (GRU MAPE=24.34%!)
- **Best Window**: 2023-10-09 (GRU MAPE=8.24%)

---

## 📝 Feature Set F2 (15 features)

**Temporal Encoding** (9):
- `is_holiday`, `hour_sin/cos`, `dow_sin/cos`, `doy_sin/cos`, `month_sin/cos`

**Historical Consumption** (2):
- `TOTAL_CONSUMPTION`, `PREMISE_COUNT`

**Weather** (4):
- `t2m_degC` (temperature)
- `tp_mm` (precipitation)
- `tcw` (total column water)
- `avg_snswrf` (net shortwave radiation)

**Note**: `avg_snlwrf` temporarily removed for compatibility with trained models.

---

## 🎯 Recommendations

### For Production
1. **Deploy TCN** for both regions
2. Add weather-based confidence intervals
3. Monitor for OOD conditions (alert at 5th/95th percentiles)
4. Consider Ottawa-specific ensemble (TCN + PatchTST)

### For Research
1. Re-evaluate OOD definition (add temporal volatility)
2. Investigate Toronto's "OOD improvement" paradox
3. Develop Ottawa-specific augmentation
4. Explore weather-aware attention mechanisms

---

## 📊 Metrics Explained

- **MAE** (Mean Absolute Error): Average prediction error in MW
  - Toronto scale: ~20-60k MW
  - Ottawa scale: ~10-50k MW

- **RMSE** (Root Mean Square Error): Penalizes large errors more
  - Similar scale to MAE but higher values

- **MAPE** (Mean Absolute Percentage Error): Scale-independent error
  - Good: <5%
  - Acceptable: 5-10%
  - Poor: >10%

- **Degradation %**: (OOD - Normal) / Normal × 100%
  - Negative = improvement
  - Positive = degradation

---

## 🐛 Known Issues

1. **PatchTST Normal Metrics Missing**: Cannot compare OOD vs normal for PatchTST
   - Models exist in forecast1111 but test_metrics.txt missing

2. **Feature Dimension Mismatch**: Original F2 had 16 features, trained models expect 15
   - Solution: Temporarily removed `avg_snlwrf`
   - Long-term: Retrain or update feature definition

3. **Toronto OOD Paradox**: Models improve on OOD windows
   - Suggests validation/test distribution mismatch
   - Needs further investigation

---

## 📞 Quick Commands

```bash
# View Toronto GRU summary
cat outputs/ood_inference/gru/Toronto_F2_fold0_summary.txt

# View Ottawa TCN summary
cat outputs/ood_inference/tcn/Ottawa_F2_fold0_summary.txt

# View overall comparison
cat outputs/ood_analysis/ood_vs_normal_summary_F2_fold0.txt

# Check specific window predictions
head outputs/ood_inference/gru/Toronto_F2_fold0_ood_predictions.csv

# View worst-case scenarios
grep "Window" outputs/ood_inference/gru/Ottawa_F2_fold0_summary.txt | sort -t: -k3 -n | tail -3
```

---

## 🔗 Related Documentation

- **Full Analysis**: `OOD_RESULTS_SUMMARY.md`
- **Setup Guide**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **Implementation Status**: `STATUS.md`
- **Package Overview**: `SUMMARY.md`

---

**Generated**: November 2024  
**Pipeline**: `exps/ood_inference/`  
**Models**: `outputs/forecast1111/per_region/`
