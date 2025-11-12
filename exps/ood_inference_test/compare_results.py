#!/usr/bin/env python3
"""
Compare Feature Sets Across Models - OOD Test Period Results
============================================================

Analyze and compare performance of different feature sets on OOD windows.
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Results directory
results_dir = Path(ROOT_DIR) / 'outputs' / 'ood_inference_test'
output_dir = results_dir / 'comparison'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Feature Set Comparison - OOD Test Period")
print("=" * 80)

# Load all results
all_results = []
models = ['gru', 'tcn', 'patchtst']
regions = ['Toronto', 'Ottawa']

for model in models:
    model_dir = results_dir / f'{model}_long_window'
    for region in regions:
        result_file = model_dir / f'ood_results_{region}.csv'
        if result_file.exists():
            df = pd.read_csv(result_file)
            all_results.append(df)
            print(f"✅ Loaded: {model.upper()} - {region}")

# Combine all results
results_df = pd.concat(all_results, ignore_index=True)
print(f"\n✅ Total results: {len(results_df)} records")

# Calculate summary statistics
summary = results_df.groupby(['model', 'region', 'feature_set'])[['mae', 'rmse', 'mape', 'smape']].mean().reset_index()
summary = summary.round(2)

print("\n" + "=" * 80)
print("Summary Statistics by Model, Region, and Feature Set")
print("=" * 80)
print(summary.to_string(index=False))

# Save summary
summary.to_csv(output_dir / 'summary_stats.csv', index=False)
print(f"\n✅ Summary saved to: {output_dir / 'summary_stats.csv'}")

# Find best configurations
print("\n" + "=" * 80)
print("Best Configurations (by MAPE)")
print("=" * 80)

for region in regions:
    print(f"\n{region}:")
    print("-" * 40)
    region_data = summary[summary['region'] == region].copy()
    region_data = region_data.sort_values('mape')
    
    for idx, row in region_data.head(5).iterrows():
        print(f"  {idx+1}. {row['model'].upper():10s} + {row['feature_set']:3s}: "
              f"MAPE={row['mape']:5.2f}%, MAE={row['mae']:8.2f}, RMSE={row['rmse']:8.2f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. MAPE comparison by feature set and model
for idx, region in enumerate(regions):
    ax = axes[0, idx]
    region_data = summary[summary['region'] == region]
    
    pivot_data = region_data.pivot(index='feature_set', columns='model', values='mape')
    pivot_data.plot(kind='bar', ax=ax)
    
    ax.set_title(f'MAPE by Feature Set - {region}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.legend(title='Model', labels=['GRU', 'TCN', 'PatchTST'])
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 2. MAE comparison by feature set and model
for idx, region in enumerate(regions):
    ax = axes[1, idx]
    region_data = summary[summary['region'] == region]
    
    pivot_data = region_data.pivot(index='feature_set', columns='model', values='mae')
    pivot_data.plot(kind='bar', ax=ax)
    
    ax.set_title(f'MAE by Feature Set - {region}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.legend(title='Model', labels=['GRU', 'TCN', 'PatchTST'])
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'feature_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_dir / 'feature_comparison.png'}")

# Create heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, region in enumerate(regions):
    ax = axes[idx]
    region_data = summary[summary['region'] == region]
    
    pivot_data = region_data.pivot(index='model', columns='feature_set', values='mape')
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'MAPE (%)'}, vmin=6, vmax=11)
    
    ax.set_title(f'MAPE Heatmap - {region}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_yticklabels(['GRU', 'TCN', 'PatchTST'], rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'mape_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Heatmap saved to: {output_dir / 'mape_heatmap.png'}")

# Generate detailed report
report_file = output_dir / 'RESULTS_SUMMARY.txt'
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("OOD INFERENCE RESULTS - TEST PERIOD (2023-03-11 to 2024-03-10)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Training Period: 2018-01-01 to 2023-03-10 (Long Window, No Validation Split)\n")
    f.write("Test Period: 2023-03-11 to 2024-03-10 (OOD Windows Only)\n\n")
    
    f.write("Models: GRU, TCN, PatchTST\n")
    f.write("Regions: Toronto, Ottawa\n")
    f.write("Feature Sets:\n")
    f.write("  - F0: IESO only (11 features)\n")
    f.write("  - F1: All features (19 features)\n")
    f.write("  - F2: Non-causally selected (15 features)\n")
    f.write("  - F3: Causally selected (14 features)\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("BEST CONFIGURATIONS (by MAPE)\n")
    f.write("=" * 80 + "\n\n")
    
    for region in regions:
        f.write(f"\n{region}:\n")
        f.write("-" * 40 + "\n")
        region_data = summary[summary['region'] == region].copy()
        region_data = region_data.sort_values('mape')
        
        for rank, (idx, row) in enumerate(region_data.head(5).iterrows(), 1):
            f.write(f"  {rank}. {row['model'].upper():10s} + {row['feature_set']:3s}: "
                   f"MAPE={row['mape']:5.2f}%, MAE={row['mae']:8.2f}, RMSE={row['rmse']:8.2f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("=" * 80 + "\n\n")
    
    # Toronto best
    toronto_best = summary[summary['region'] == 'Toronto'].sort_values('mape').iloc[0]
    f.write(f"Toronto Best: {toronto_best['model'].upper()} + {toronto_best['feature_set']} "
            f"(MAPE: {toronto_best['mape']:.2f}%)\n")
    
    # Ottawa best
    ottawa_best = summary[summary['region'] == 'Ottawa'].sort_values('mape').iloc[0]
    f.write(f"Ottawa Best: {ottawa_best['model'].upper()} + {ottawa_best['feature_set']} "
            f"(MAPE: {ottawa_best['mape']:.2f}%)\n\n")
    
    # Best feature set per model
    f.write("Best Feature Set per Model:\n")
    for model in models:
        model_data = summary[summary['model'] == model].sort_values('mape')
        best = model_data.iloc[0]
        f.write(f"  {model.upper():10s}: {best['feature_set']} "
                f"({best['region']}, MAPE: {best['mape']:.2f}%)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("PERFORMANCE RANKING BY FEATURE SET\n")
    f.write("=" * 80 + "\n\n")
    
    for region in regions:
        f.write(f"\n{region}:\n")
        region_data = summary[summary['region'] == region].groupby('feature_set')['mape'].mean().sort_values()
        for fs, mape in region_data.items():
            f.write(f"  {fs}: {mape:.2f}%\n")

print(f"\n✅ Detailed report saved to: {report_file}")

print("\n" + "=" * 80)
print("✅ Analysis Complete!")
print("=" * 80)
print(f"\nAll outputs saved to: {output_dir}")
print("\nFiles generated:")
print(f"  - summary_stats.csv")
print(f"  - feature_comparison.png")
print(f"  - mape_heatmap.png")
print(f"  - RESULTS_SUMMARY.txt")
print("=" * 80)
