#!/usr/bin/env python3
"""
Compare OOD Performance Across Feature Sets
============================================

This script compares the OOD performance of different feature sets (F0, F1, F2, F3)
for all models and regions.

Usage:
    python compare_feature_sets.py [--fold FOLD]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

def load_ood_metrics(model, region, feature_set, fold=0):
    """Load OOD metrics for a specific configuration."""
    base_dir = ROOT_DIR / 'outputs' / 'ood_inference' / model
    metrics_file = base_dir / f'{region}_{feature_set}_fold{fold}_ood_metrics.csv'
    
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    return {
        'mae_mean': df['MAE'].mean(),
        'mae_std': df['MAE'].std(),
        'rmse_mean': df['RMSE'].mean(),
        'rmse_std': df['RMSE'].std(),
        'mape_mean': df['MAPE'].mean(),
        'mape_std': df['MAPE'].std(),
        'n_windows': len(df)
    }

def create_comparison_table(models, regions, feature_sets, fold=0):
    """Create a comparison table of all configurations."""
    results = []
    
    for feature_set in feature_sets:
        for region in regions:
            for model in models:
                metrics = load_ood_metrics(model, region, feature_set, fold)
                if metrics:
                    results.append({
                        'Feature_Set': feature_set,
                        'Region': region,
                        'Model': model,
                        'MAE': metrics['mae_mean'],
                        'MAE_std': metrics['mae_std'],
                        'RMSE': metrics['rmse_mean'],
                        'RMSE_std': metrics['rmse_std'],
                        'MAPE': metrics['mape_mean'],
                        'MAPE_std': metrics['mape_std'],
                        'N_Windows': metrics['n_windows']
                    })
    
    return pd.DataFrame(results)

def find_best_configurations(df):
    """Find best feature set for each model-region combination."""
    best_configs = []
    
    for region in df['Region'].unique():
        for model in df['Model'].unique():
            subset = df[(df['Region'] == region) & (df['Model'] == model)]
            if len(subset) == 0:
                continue
            
            best_idx = subset['MAPE'].idxmin()
            best = subset.loc[best_idx]
            
            best_configs.append({
                'Region': region,
                'Model': model,
                'Best_Feature_Set': best['Feature_Set'],
                'Best_MAPE': best['MAPE'],
                'MAPE_std': best['MAPE_std']
            })
    
    return pd.DataFrame(best_configs)

def create_comparison_plots(df, output_dir):
    """Create visualization comparing feature sets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # 1. MAPE comparison by feature set (grouped bar chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OOD Performance Comparison Across Feature Sets', fontsize=16, fontweight='bold')
    
    regions = df['Region'].unique()
    models = ['gru', 'tcn', 'patchtst']
    
    for idx, (region, model) in enumerate([(r, m) for r in regions for m in models]):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        subset = df[(df['Region'] == region) & (df['Model'] == model.upper())]
        
        if len(subset) > 0:
            x = np.arange(len(subset))
            bars = ax.bar(x, subset['MAPE'], yerr=subset['MAPE_std'], 
                         capsize=5, alpha=0.7, color=sns.color_palette("husl", len(subset)))
            ax.set_xticks(x)
            ax.set_xticklabels(subset['Feature_Set'])
            ax.set_ylabel('MAPE (%)', fontsize=10)
            ax.set_title(f'{region} - {model.upper()}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, subset['MAPE']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{region} - {model.upper()} (No data)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_sets_comparison_mape.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature set ranking heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Set Performance Ranking (Lower MAPE = Better)', fontsize=14, fontweight='bold')
    
    for idx, region in enumerate(regions):
        subset = df[df['Region'] == region]
        pivot = subset.pivot(index='Model', columns='Feature_Set', values='MAPE')
        
        # Create ranking (1 = best, 4 = worst)
        ranking = pivot.rank(axis=1)
        
        sns.heatmap(ranking, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Rank (1=Best)'}, ax=axes[idx],
                   linewidths=0.5, linecolor='gray')
        axes[idx].set_title(f'{region}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Model', fontsize=10)
        axes[idx].set_xlabel('Feature Set', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_sets_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Best feature set per model-region
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_configs = find_best_configurations(df)
    
    # Create grouped bar chart
    models_list = ['GRU', 'TCN', 'PATCHTST']
    x = np.arange(len(models_list))
    width = 0.35
    
    toronto_data = []
    ottawa_data = []
    
    for model in models_list:
        toronto_best = best_configs[(best_configs['Region'] == 'Toronto') & 
                                    (best_configs['Model'] == model)]
        ottawa_best = best_configs[(best_configs['Region'] == 'Ottawa') & 
                                   (best_configs['Model'] == model)]
        
        toronto_data.append(toronto_best['Best_MAPE'].values[0] if len(toronto_best) > 0 else 0)
        ottawa_data.append(ottawa_best['Best_MAPE'].values[0] if len(ottawa_best) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, toronto_data, width, label='Toronto', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, ottawa_data, width, label='Ottawa', alpha=0.8, color='coral')
    
    ax.set_ylabel('Best MAPE (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Best OOD Performance by Model (Using Best Feature Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_feature_set_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")

def generate_summary_report(df, best_configs, output_file):
    """Generate a text summary report."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Feature Set Comparison Summary\n")
        f.write("="*80 + "\n\n")
        
        # Best feature sets
        f.write("Best Feature Set for Each Model-Region Combination:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Region':<12} {'Model':<12} {'Best Feature':<15} {'MAPE':<12} {'Std Dev':<12}\n")
        f.write("-"*80 + "\n")
        for _, row in best_configs.iterrows():
            f.write(f"{row['Region']:<12} {row['Model']:<12} "
                   f"{row['Best_Feature_Set']:<15} "
                   f"{row['Best_MAPE']:>6.2f}%     ±{row['MAPE_std']:>5.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed Performance Comparison\n")
        f.write("="*80 + "\n\n")
        
        for region in df['Region'].unique():
            f.write(f"\n{region}\n")
            f.write("-"*80 + "\n")
            
            region_df = df[df['Region'] == region]
            
            for model in ['GRU', 'TCN', 'PATCHTST']:
                model_df = region_df[region_df['Model'] == model]
                if len(model_df) == 0:
                    continue
                
                f.write(f"\n{model}:\n")
                for _, row in model_df.iterrows():
                    f.write(f"  {row['Feature_Set']}: MAPE={row['MAPE']:.2f}% (±{row['MAPE_std']:.2f}%), "
                           f"MAE={row['MAE']:.0f} MW (±{row['MAE_std']:.0f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Key Insights\n")
        f.write("="*80 + "\n\n")
        
        # Calculate average MAPE per feature set
        f.write("Average MAPE by Feature Set (across all models and regions):\n")
        for fs in df['Feature_Set'].unique():
            avg_mape = df[df['Feature_Set'] == fs]['MAPE'].mean()
            f.write(f"  {fs}: {avg_mape:.2f}%\n")
        
        # Find overall best feature set
        fs_avg = df.groupby('Feature_Set')['MAPE'].mean()
        best_fs = fs_avg.idxmin()
        f.write(f"\n🏆 Overall Best Feature Set: {best_fs} (Average MAPE: {fs_avg[best_fs]:.2f}%)\n")
        
        # Model-specific insights
        f.write("\nModel-Specific Best Feature Sets:\n")
        for model in ['GRU', 'TCN', 'PATCHTST']:
            model_df = df[df['Model'] == model]
            if len(model_df) > 0:
                best_fs_model = model_df.groupby('Feature_Set')['MAPE'].mean().idxmin()
                avg_mape_model = model_df.groupby('Feature_Set')['MAPE'].mean()[best_fs_model]
                f.write(f"  {model}: {best_fs_model} (Average MAPE: {avg_mape_model:.2f}%)\n")
    
    print(f"✅ Summary report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare OOD performance across feature sets')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (default: 0)')
    args = parser.parse_args()
    
    # Configuration
    models = ['gru', 'tcn', 'patchtst']
    regions = ['Toronto', 'Ottawa']
    feature_sets = ['F0', 'F1', 'F2', 'F3']
    
    print("="*80)
    print("Feature Set Comparison Analysis")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Regions: {', '.join(regions)}")
    print(f"Feature Sets: {', '.join(feature_sets)}")
    print(f"Fold: {args.fold}")
    print("="*80)
    print()
    
    # Load all metrics
    print("Loading OOD metrics...")
    df = create_comparison_table(models, regions, feature_sets, args.fold)
    
    if len(df) == 0:
        print("❌ No data found. Make sure OOD inference has been run for all feature sets.")
        return
    
    print(f"✅ Loaded {len(df)} configurations")
    print()
    
    # Find best configurations
    print("Finding best feature sets...")
    best_configs = find_best_configurations(df)
    print(f"✅ Found best configurations for {len(best_configs)} model-region pairs")
    print()
    
    # Output directory
    output_dir = ROOT_DIR / 'outputs' / 'ood_analysis' / 'feature_set_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparison CSV
    csv_file = output_dir / f'feature_sets_comparison_fold{args.fold}.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Detailed comparison saved to {csv_file}")
    
    # Save best configurations CSV
    best_csv = output_dir / f'best_feature_sets_fold{args.fold}.csv'
    best_configs.to_csv(best_csv, index=False)
    print(f"✅ Best configurations saved to {best_csv}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_comparison_plots(df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    summary_file = output_dir / f'feature_sets_summary_fold{args.fold}.txt'
    generate_summary_report(df, best_configs, summary_file)
    
    print()
    print("="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved in: {output_dir}/")
    print("  - feature_sets_comparison_fold0.csv")
    print("  - best_feature_sets_fold0.csv")
    print("  - feature_sets_summary_fold0.txt")
    print("  - feature_sets_comparison_mape.png")
    print("  - feature_sets_ranking_heatmap.png")
    print("  - best_feature_set_performance.png")
    print()

if __name__ == '__main__':
    main()
