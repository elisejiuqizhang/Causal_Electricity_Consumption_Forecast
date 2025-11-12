"""
Compare OOD performance with normal test performance
Analyze degradation in model performance under extreme weather conditions
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

def load_normal_performance(model_type, region, feature_set, seed, fold):
    """Load normal test performance from training results"""
    model_dirs = {
        'gru': 'gru_single_train',
        'tcn': 'tcn_single_train',
        'patchtst': 'patchtst_single_train'
    }
    
    training_config = "bs64_ep500_lr0.0001_tr0.93_vr0.07_pat20_esep0.0001"
    
    if model_type == 'patchtst':
        # PatchTST has a more complex path structure
        model_path = os.path.join(
            ROOT_DIR, 'outputs', 'forecast1111', 'per_region', model_dirs[model_type],
            region.replace(" ", "_"),
            "dm32_nh4_nl3_pl16_ps8_inlen168_h24_scalerstandard",
            "non_overlap",
            training_config,
            feature_set,
            str(seed),
            f'fold_{fold}'
        )
    else:
        model_path = os.path.join(
            ROOT_DIR, 'outputs', 'forecast1111', 'per_region', model_dirs[model_type],
            region.replace(" ", "_"),
            training_config,
            feature_set,
            str(seed),
            f'fold_{fold}'
        )
    
    # Load test metrics
    metrics_file = os.path.join(model_path, 'test_metrics.txt')
    
    if not os.path.exists(metrics_file):
        return None
    
    # Parse metrics file
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            if 'MAE' in line:
                metrics['MAE'] = float(line.split(':')[1].strip())
            elif 'RMSE' in line:
                metrics['RMSE'] = float(line.split(':')[1].strip())
            elif 'MAPE' in line:
                metrics['MAPE'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'SMAPE' in line:
                metrics['SMAPE'] = float(line.split(':')[1].strip().replace('%', ''))
    
    return metrics


def load_ood_performance(model_type, region, feature_set, fold):
    """Load OOD inference results"""
    ood_dir = os.path.join(ROOT_DIR, 'outputs', 'ood_inference', model_type)
    
    metrics_file = os.path.join(ood_dir, f'{region}_{feature_set}_fold{fold}_ood_metrics.csv')
    
    if not os.path.exists(metrics_file):
        return None
    
    ood_df = pd.read_csv(metrics_file)
    
    # Calculate average metrics
    metrics = {
        'MAE': ood_df['MAE'].mean(),
        'RMSE': ood_df['RMSE'].mean(),
        'MAPE': ood_df['MAPE'].mean(),
        'SMAPE': ood_df['SMAPE'].mean(),
        'MAE_std': ood_df['MAE'].std(),
        'RMSE_std': ood_df['RMSE'].std(),
        'MAPE_std': ood_df['MAPE'].std(),
        'SMAPE_std': ood_df['SMAPE'].std(),
        'n_windows': len(ood_df)
    }
    
    return metrics, ood_df


def main():
    parser = argparse.ArgumentParser(description='Compare OOD vs Normal Performance')
    
    parser.add_argument('--regions', nargs='+', default=['Toronto', 'Ottawa'],
                       help='Regions to analyze')
    parser.add_argument('--models', nargs='+', default=['gru', 'tcn', 'patchtst'],
                       help='Models to compare')
    parser.add_argument('--feature_set', type=str, default='F2',
                       help='Feature set')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number')
    parser.add_argument('--seeds', nargs='+', type=int, default=[97, 97, 597],
                       help='Seeds for each model (gru, tcn, patchtst)')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(ROOT_DIR, 'outputs', 'ood_analysis'),
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map models to seeds
    seed_map = dict(zip(args.models, args.seeds))
    
    # Collect all results
    results = []
    
    for region in args.regions:
        for model in args.models:
            seed = seed_map[model]
            
            print(f"\nProcessing {model.upper()} - {region}")
            
            # Load normal performance
            normal_metrics = load_normal_performance(
                model, region, args.feature_set, seed, args.fold
            )
            
            # Load OOD performance
            ood_result = load_ood_performance(
                model, region, args.feature_set, args.fold
            )
            
            if normal_metrics is None:
                print(f"  Warning: Could not load normal metrics")
                continue
                
            if ood_result is None:
                print(f"  Warning: Could not load OOD metrics")
                continue
            
            ood_metrics, ood_df = ood_result
            
            # Calculate degradation
            mae_degradation = ((ood_metrics['MAE'] - normal_metrics['MAE']) / normal_metrics['MAE']) * 100
            rmse_degradation = ((ood_metrics['RMSE'] - normal_metrics['RMSE']) / normal_metrics['RMSE']) * 100
            mape_degradation = ((ood_metrics['MAPE'] - normal_metrics['MAPE']) / normal_metrics['MAPE']) * 100
            
            print(f"  Normal MAE: {normal_metrics['MAE']:.2f}, OOD MAE: {ood_metrics['MAE']:.2f} ({mae_degradation:+.1f}%)")
            print(f"  Normal RMSE: {normal_metrics['RMSE']:.2f}, OOD RMSE: {ood_metrics['RMSE']:.2f} ({rmse_degradation:+.1f}%)")
            print(f"  Normal MAPE: {normal_metrics['MAPE']:.2f}%, OOD MAPE: {ood_metrics['MAPE']:.2f}% ({mape_degradation:+.1f}%)")
            
            results.append({
                'Model': model.upper(),
                'Region': region,
                'Normal_MAE': normal_metrics['MAE'],
                'OOD_MAE': ood_metrics['MAE'],
                'OOD_MAE_std': ood_metrics['MAE_std'],
                'MAE_Degradation_%': mae_degradation,
                'Normal_RMSE': normal_metrics['RMSE'],
                'OOD_RMSE': ood_metrics['RMSE'],
                'OOD_RMSE_std': ood_metrics['RMSE_std'],
                'RMSE_Degradation_%': rmse_degradation,
                'Normal_MAPE': normal_metrics['MAPE'],
                'OOD_MAPE': ood_metrics['MAPE'],
                'OOD_MAPE_std': ood_metrics['MAPE_std'],
                'MAPE_Degradation_%': mape_degradation,
                'N_OOD_Windows': ood_metrics['n_windows']
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(args.output_dir, f'ood_vs_normal_comparison_{args.feature_set}_fold{args.fold}.csv')
    results_df.to_csv(output_file, index=False)
    
    # Create summary report
    summary_file = os.path.join(args.output_dir, f'ood_vs_normal_summary_{args.feature_set}_fold{args.fold}.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OOD vs Normal Performance Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Regions: {', '.join(args.regions)}\n")
        f.write(f"Models: {', '.join([m.upper() for m in args.models])}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Overall Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            avg_degradation = results_df[f'{metric}_Degradation_%'].mean()
            f.write(f"Average {metric} Degradation: {avg_degradation:+.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Model Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for model in args.models:
            model_df = results_df[results_df['Model'] == model.upper()]
            if len(model_df) > 0:
                f.write(f"\n{model.upper()}:\n")
                f.write(f"  Average MAE Degradation: {model_df['MAE_Degradation_%'].mean():+.2f}%\n")
                f.write(f"  Average RMSE Degradation: {model_df['RMSE_Degradation_%'].mean():+.2f}%\n")
                f.write(f"  Average MAPE Degradation: {model_df['MAPE_Degradation_%'].mean():+.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Region Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for region in args.regions:
            region_df = results_df[results_df['Region'] == region]
            if len(region_df) > 0:
                f.write(f"\n{region}:\n")
                f.write(f"  Average MAE Degradation: {region_df['MAE_Degradation_%'].mean():+.2f}%\n")
                f.write(f"  Average RMSE Degradation: {region_df['RMSE_Degradation_%'].mean():+.2f}%\n")
                f.write(f"  Average MAPE Degradation: {region_df['MAPE_Degradation_%'].mean():+.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(results_df.to_string(index=False))
    
    # Create visualization
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['MAE', 'RMSE', 'MAPE']
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for grouped bar chart
            data = results_df.pivot(index='Model', columns='Region', values=f'{metric}_Degradation_%')
            
            data.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} Degradation (%)')
            ax.set_ylabel('Degradation (%)')
            ax.set_xlabel('Model')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.legend(title='Region')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(args.output_dir, f'ood_degradation_comparison_{args.feature_set}_fold{args.fold}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*80}")
        print("Summary")
        print('='*80)
        print(f"\nResults saved to:")
        print(f"  - {output_file}")
        print(f"  - {summary_file}")
        print(f"  - {plot_file}")
        print(f"\nOverall average degradation:")
        for metric in ['MAE', 'RMSE', 'MAPE']:
            avg_deg = results_df[f'{metric}_Degradation_%'].mean()
            print(f"  {metric}: {avg_deg:+.2f}%")


if __name__ == '__main__':
    main()
