#!/usr/bin/env python3
"""
Identify Out-Of-Distribution (OOD) weather windows for inference testing.

This script:
1. Loads weather data (temperature, precipitation)
2. Calculates 5th and 95th percentiles for training data
3. Identifies 24-hour windows with OOD conditions
4. Saves a report for subsequent inference tasks

Usage:
    python identify_ood_weather.py --region Toronto --data_dir data/ieso_era5
"""

import pandas as pd
import numpy as np
import argparse
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import your project's utilities
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'data_utils'))
# from info_features import get_feature_columns  # Not needed for this script


def load_weather_data(data_dir, region, data_file_prefix='combined_ieso_era5_avg'):
    """Load weather data for a specific region"""
    filename = f"{data_file_prefix}_{region.lower()}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert datetime column (check multiple possible names)
    datetime_col = None
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(dt in col_lower for dt in ['datetime', 'time', 'date']):
            datetime_col = col
            break
    
    if datetime_col is None:
        raise ValueError(f"No datetime column found. Available columns: {list(df.columns)}")
    
    print(f"Using datetime column: '{datetime_col}'")
    df['datetime'] = pd.to_datetime(df[datetime_col])
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def split_data_by_date(df, train_ratio=0.93, val_ratio=0.07):
    """Split data into train/val/test sets by time"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nData split:")
    print(f"  Train: {train_df['datetime'].min()} to {train_df['datetime'].max()} ({len(train_df)} samples)")
    print(f"  Val:   {val_df['datetime'].min()} to {val_df['datetime'].max()} ({len(val_df)} samples)")
    print(f"  Test:  {test_df['datetime'].min()} to {test_df['datetime'].max()} ({len(test_df)} samples)")
    
    return train_df, val_df, test_df


def calculate_percentiles(train_df, weather_vars=['temperature_2m', 't2m', 'precipitation', 'tp']):
    """Calculate 5th and 95th percentiles for weather variables from training data"""
    
    # Find which weather columns exist
    available_vars = [col for col in weather_vars if col in train_df.columns]
    
    if not available_vars:
        # Try to find temperature and precipitation columns
        temp_cols = [col for col in train_df.columns if 't2m' in col.lower() or 'temp' in col.lower()]
        precip_cols = [col for col in train_df.columns if 'tp' in col.lower() or 'precip' in col.lower() or 'rain' in col.lower()]
        available_vars = temp_cols + precip_cols
    
    if not available_vars:
        raise ValueError(f"No weather variables found in data. Columns: {list(train_df.columns)}")
    
    print(f"\nWeather variables found: {available_vars}")
    
    percentiles = {}
    for var in available_vars:
        p5 = train_df[var].quantile(0.05)
        p95 = train_df[var].quantile(0.95)
        percentiles[var] = {
            'p5': p5,
            'p95': p95,
            'mean': train_df[var].mean(),
            'std': train_df[var].std(),
            'min': train_df[var].min(),
            'max': train_df[var].max()
        }
        print(f"\n{var}:")
        print(f"  5th percentile:  {p5:.4f}")
        print(f"  95th percentile: {p95:.4f}")
        print(f"  Mean:            {percentiles[var]['mean']:.4f}")
        print(f"  Std:             {percentiles[var]['std']:.4f}")
        print(f"  Range:           [{percentiles[var]['min']:.4f}, {percentiles[var]['max']:.4f}]")
    
    return percentiles, available_vars


def identify_ood_windows(df, percentiles, weather_vars, window_size=24, threshold=0.5):
    """
    Identify 24-hour windows with OOD weather conditions.
    
    A window is considered OOD if:
    - At least threshold fraction (e.g., 50%) of hours have at least one weather variable
      outside the [p5, p95] range
    
    Args:
        df: DataFrame with weather data
        percentiles: Dictionary with percentile thresholds
        weather_vars: List of weather variable names
        window_size: Size of window in hours (default: 24)
        threshold: Fraction of hours that must be OOD (default: 0.5)
    
    Returns:
        DataFrame with OOD windows and their statistics
    """
    
    n = len(df)
    ood_windows = []
    
    print(f"\nSearching for OOD windows...")
    print(f"  Window size: {window_size} hours")
    print(f"  Threshold: {threshold*100}% of hours must have OOD conditions")
    
    # Create OOD flags for each variable
    ood_flags = {}
    for var in weather_vars:
        p5 = percentiles[var]['p5']
        p95 = percentiles[var]['p95']
        # Flag: 1 if below p5, 2 if above p95, 0 if in normal range
        ood_flags[var] = np.where(df[var] < p5, -1,
                                   np.where(df[var] > p95, 1, 0))
    
    # Slide window across data
    for i in range(n - window_size + 1):
        window_data = df.iloc[i:i+window_size]
        
        # Count hours with OOD conditions
        ood_hours = 0
        var_ood_counts = {var: {'below': 0, 'above': 0} for var in weather_vars}
        
        for j in range(window_size):
            is_ood = False
            for var in weather_vars:
                flag = ood_flags[var][i+j]
                if flag != 0:
                    is_ood = True
                    if flag < 0:
                        var_ood_counts[var]['below'] += 1
                    else:
                        var_ood_counts[var]['above'] += 1
            
            if is_ood:
                ood_hours += 1
        
        # Check if window meets OOD threshold
        ood_fraction = ood_hours / window_size
        
        if ood_fraction >= threshold:
            # Calculate statistics for this window
            window_stats = {
                'start_datetime': window_data['datetime'].iloc[0],
                'end_datetime': window_data['datetime'].iloc[-1],
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'ood_hours': ood_hours,
                'ood_fraction': ood_fraction
            }
            
            # Add per-variable statistics
            for var in weather_vars:
                window_stats[f'{var}_mean'] = window_data[var].mean()
                window_stats[f'{var}_min'] = window_data[var].min()
                window_stats[f'{var}_max'] = window_data[var].max()
                window_stats[f'{var}_below_p5'] = var_ood_counts[var]['below']
                window_stats[f'{var}_above_p95'] = var_ood_counts[var]['above']
            
            ood_windows.append(window_stats)
    
    # Convert to DataFrame
    ood_df = pd.DataFrame(ood_windows)
    
    print(f"\nFound {len(ood_df)} OOD windows")
    
    return ood_df


def remove_overlapping_windows(ood_df, min_gap_hours=24):
    """
    Remove overlapping windows, keeping the most extreme ones.
    Keep windows separated by at least min_gap_hours.
    """
    if len(ood_df) == 0:
        return ood_df
    
    # Sort by OOD fraction (most extreme first)
    ood_df = ood_df.sort_values('ood_fraction', ascending=False).reset_index(drop=True)
    
    selected = []
    selected_indices = set()
    
    for idx, row in ood_df.iterrows():
        start_idx = row['start_idx']
        end_idx = row['end_idx']
        
        # Check overlap with already selected windows
        overlap = False
        for sel_start, sel_end in selected_indices:
            # Check if there's overlap or too close
            if not (end_idx + min_gap_hours < sel_start or start_idx > sel_end + min_gap_hours):
                overlap = True
                break
        
        if not overlap:
            selected.append(row)
            selected_indices.add((start_idx, end_idx))
    
    result_df = pd.DataFrame(selected).reset_index(drop=True)
    print(f"After removing overlaps (min gap: {min_gap_hours}h): {len(result_df)} windows")
    
    return result_df


def create_ood_report(ood_df, percentiles, weather_vars, output_dir, region, split='test'):
    """Create detailed OOD report and visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save OOD windows to CSV
    csv_path = os.path.join(output_dir, f'ood_windows_{region}_{split}.csv')
    ood_df.to_csv(csv_path, index=False)
    print(f"\nSaved OOD windows to: {csv_path}")
    
    # Create text report
    report_path = os.path.join(output_dir, f'ood_report_{region}_{split}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Out-of-Distribution Weather Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Region: {region}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Training Data Percentiles:\n")
        f.write(f"{'-'*80}\n")
        for var in weather_vars:
            f.write(f"\n{var}:\n")
            f.write(f"  5th percentile:  {percentiles[var]['p5']:.4f}\n")
            f.write(f"  95th percentile: {percentiles[var]['p95']:.4f}\n")
            f.write(f"  Mean ± Std:      {percentiles[var]['mean']:.4f} ± {percentiles[var]['std']:.4f}\n")
            f.write(f"  Range:           [{percentiles[var]['min']:.4f}, {percentiles[var]['max']:.4f}]\n")
        
        f.write(f"\n\nOOD Windows Summary:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Total OOD windows found: {len(ood_df)}\n\n")
        
        if len(ood_df) > 0:
            f.write(f"Top 10 Most Extreme OOD Windows:\n")
            f.write(f"{'-'*80}\n\n")
            
            top_windows = ood_df.nlargest(min(10, len(ood_df)), 'ood_fraction')
            
            for idx, row in top_windows.iterrows():
                f.write(f"Window #{idx+1}:\n")
                f.write(f"  Time range: {row['start_datetime']} to {row['end_datetime']}\n")
                f.write(f"  Index range: [{row['start_idx']}, {row['end_idx']}]\n")
                f.write(f"  OOD fraction: {row['ood_fraction']*100:.1f}% ({row['ood_hours']}/24 hours)\n")
                
                for var in weather_vars:
                    f.write(f"  {var}:\n")
                    f.write(f"    Mean: {row[f'{var}_mean']:.4f} (normal: {percentiles[var]['mean']:.4f})\n")
                    f.write(f"    Range: [{row[f'{var}_min']:.4f}, {row[f'{var}_max']:.4f}]\n")
                    f.write(f"    Below p5: {row[f'{var}_below_p5']} hours, Above p95: {row[f'{var}_above_p95']} hours\n")
                
                f.write(f"\n")
    
    print(f"Saved report to: {report_path}")
    
    # Create visualization
    if len(ood_df) > 0:
        plot_ood_distribution(ood_df, percentiles, weather_vars, output_dir, region, split)


def plot_ood_distribution(ood_df, percentiles, weather_vars, output_dir, region, split):
    """Create visualizations of OOD windows"""
    
    n_vars = len(weather_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var in enumerate(weather_vars):
        ax = axes[idx]
        
        # Plot histogram of mean values in OOD windows
        ax.hist(ood_df[f'{var}_mean'], bins=30, alpha=0.6, label='OOD windows', edgecolor='black')
        
        # Add reference lines
        p5 = percentiles[var]['p5']
        p95 = percentiles[var]['p95']
        mean = percentiles[var]['mean']
        
        ax.axvline(p5, color='blue', linestyle='--', linewidth=2, label=f'5th percentile ({p5:.2f})')
        ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'95th percentile ({p95:.2f})')
        ax.axvline(mean, color='green', linestyle='-', linewidth=2, label=f'Training mean ({mean:.2f})')
        
        ax.set_xlabel(f'{var}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {var} in OOD Windows')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'ood_distribution_{region}_{split}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Identify OOD weather windows for inference testing')
    parser.add_argument('--region', type=str, default='Toronto',
                        help='Region name (e.g., Toronto, Peel, Hamilton)')
    parser.add_argument('--data_dir', type=str, default='data/ieso_era5',
                        help='Directory containing weather data')
    parser.add_argument('--data_file_prefix', type=str, default='combined_ieso_era5_avg',
                        help='Prefix for data files')
    parser.add_argument('--train_ratio', type=float, default=0.93,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.07,
                        help='Validation data ratio')
    parser.add_argument('--window_size', type=int, default=24,
                        help='Window size in hours (default: 24)')
    parser.add_argument('--ood_threshold', type=float, default=0.5,
                        help='Fraction of hours that must be OOD (default: 0.5)')
    parser.add_argument('--min_gap', type=int, default=24,
                        help='Minimum gap between OOD windows in hours (default: 24)')
    parser.add_argument('--output_dir', type=str, default='outputs/ood_analysis',
                        help='Output directory for reports')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'],
                        help='Which data split to analyze (default: test)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OOD Weather Window Identification")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Region: {args.region}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Window size: {args.window_size} hours")
    print(f"  OOD threshold: {args.ood_threshold*100}%")
    print(f"  Min gap: {args.min_gap} hours")
    print(f"  Split: {args.split}")
    
    # Load data
    df = load_weather_data(args.data_dir, args.region, args.data_file_prefix)
    
    # Split data
    train_df, val_df, test_df = split_data_by_date(df, args.train_ratio, args.val_ratio)
    
    # Calculate percentiles from training data
    percentiles, weather_vars = calculate_percentiles(train_df)
    
    # Determine which split(s) to analyze
    splits_to_analyze = []
    if args.split == 'all':
        splits_to_analyze = [('train', train_df), ('val', val_df), ('test', test_df)]
    elif args.split == 'train':
        splits_to_analyze = [('train', train_df)]
    elif args.split == 'val':
        splits_to_analyze = [('val', val_df)]
    else:  # test
        splits_to_analyze = [('test', test_df)]
    
    # Analyze each split
    for split_name, split_df in splits_to_analyze:
        print(f"\n{'='*80}")
        print(f"Analyzing {split_name} split")
        print(f"{'='*80}")
        
        # Identify OOD windows
        ood_df = identify_ood_windows(split_df, percentiles, weather_vars, 
                                       args.window_size, args.ood_threshold)
        
        # Remove overlapping windows
        if len(ood_df) > 0:
            ood_df = remove_overlapping_windows(ood_df, args.min_gap)
        
        # Create report
        create_ood_report(ood_df, percentiles, weather_vars, args.output_dir, 
                         args.region, split_name)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
