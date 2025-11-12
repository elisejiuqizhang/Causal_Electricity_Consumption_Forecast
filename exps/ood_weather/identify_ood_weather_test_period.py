#!/usr/bin/env python3
"""
Identify OOD Weather Windows in Test Period (2023-03-11 to 2024-03-10)
=======================================================================

This script identifies out-of-distribution (OOD) weather conditions in the test period
based on the training period distribution (2018-01-01 to 2023-03-10).

Usage:
    python identify_ood_weather_test_period.py --region REGION_NAME
    
Example:
    python identify_ood_weather_test_period.py --region Toronto
    python identify_ood_weather_test_period.py --region Ottawa
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_utils.info_cities import dict_regions

# Create city name mapping from dict_regions
city_name_mapping = {}
for region, cities in dict_regions.items():
    if len(cities) == 1:
        # Single city regions: Toronto, Hamilton, etc.
        city_name_mapping[region] = cities[0].lower()
    else:
        # Multi-city regions: use average from first city
        city_name_mapping[region] = cities[0].lower()

# Period definitions
TRAIN_START = '2018-01-01'
TRAIN_END = '2023-03-10'
TEST_START = '2023-03-11'
TEST_END = '2024-03-10'

# Weather features to consider for OOD detection
# Only temperature and precipitation
WEATHER_FEATURES = ['t2m_degC', 'tp_mm']

def load_data(region):
    """Load data for the specified region."""
    print(f"Loading data for {region}...")
    
    # Map region name to city name
    city_name = city_name_mapping.get(region, region)
    
    # Construct file path
    data_dir = ROOT_DIR / 'data' / 'ieso_era5'
    data_file = data_dir / f'combined_ieso_era5_avg_{city_name.lower()}.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Convert datetime column (handle both 'time' and 'datetime_utc' column names)
    if 'time' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
    else:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    df = df.sort_values('datetime_utc').reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
    
    return df

def split_data(df):
    """Split data into training and test periods."""
    train_df = df[(df['datetime_utc'] >= TRAIN_START) & 
                  (df['datetime_utc'] <= TRAIN_END)].copy()
    test_df = df[(df['datetime_utc'] >= TEST_START) & 
                 (df['datetime_utc'] <= TEST_END)].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {TRAIN_START} to {TRAIN_END}")
    print(f"     → {len(train_df)} records ({len(train_df)/24:.0f} days)")
    print(f"   Test: {TEST_START} to {TEST_END}")
    print(f"     → {len(test_df)} records ({len(test_df)/24:.0f} days)")
    
    return train_df, test_df

def calculate_percentiles(train_df, features, lower_pct=5, upper_pct=95):
    """Calculate percentile thresholds from training data."""
    percentiles = {}
    
    print(f"\n🔍 Calculating {lower_pct}th/{upper_pct}th percentiles from training data...")
    print(f"{'Feature':<20} {'5th %ile':<12} {'95th %ile':<12} {'Train Mean':<12}")
    print("-" * 60)
    
    for feat in features:
        if feat not in train_df.columns:
            print(f"⚠️  Feature {feat} not found in data, skipping...")
            continue
        
        lower = np.percentile(train_df[feat].dropna(), lower_pct)
        upper = np.percentile(train_df[feat].dropna(), upper_pct)
        mean = train_df[feat].mean()
        
        percentiles[feat] = {'lower': lower, 'upper': upper, 'mean': mean}
        print(f"{feat:<20} {lower:<12.4f} {upper:<12.4f} {mean:<12.4f}")
    
    return percentiles

def identify_ood_timesteps(test_df, percentiles):
    """Identify timesteps that are outside percentile ranges."""
    ood_flags = pd.DataFrame(index=test_df.index)
    
    for feat, thresholds in percentiles.items():
        if feat not in test_df.columns:
            continue
        
        # Mark timesteps outside [lower, upper] range
        below = test_df[feat] < thresholds['lower']
        above = test_df[feat] > thresholds['upper']
        ood_flags[f'{feat}_ood'] = below | above
    
    # Overall OOD: if ANY feature is OOD
    ood_flags['any_ood'] = ood_flags.any(axis=1)
    
    # Count how many features are OOD at each timestep
    ood_flags['n_ood_features'] = ood_flags[[c for c in ood_flags.columns 
                                             if c.endswith('_ood') and c != 'any_ood']].sum(axis=1)
    
    return ood_flags

def identify_ood_windows(test_df, ood_flags, window_size=24, ood_threshold=0.5):
    """
    Identify contiguous windows where a significant portion is OOD.
    
    Args:
        test_df: Test dataframe
        ood_flags: Boolean flags for OOD timesteps
        window_size: Size of window in hours (default: 24 = 1 day)
        ood_threshold: Fraction of window that must be OOD (default: 0.5 = 50%)
    """
    print(f"\n🔍 Identifying OOD windows (size={window_size}h, threshold={ood_threshold*100:.0f}%)...")
    
    ood_windows = []
    
    # Slide window across test period
    for i in range(len(test_df) - window_size + 1):
        window_slice = slice(i, i + window_size)
        window_ood_flags = ood_flags.iloc[window_slice]
        
        # Calculate fraction of OOD timesteps in window
        ood_fraction = window_ood_flags['any_ood'].mean()
        
        if ood_fraction >= ood_threshold:
            start_idx = test_df.index[i]
            end_idx = test_df.index[i + window_size - 1]
            
            ood_windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_datetime': test_df.loc[start_idx, 'datetime_utc'],
                'end_datetime': test_df.loc[end_idx, 'datetime_utc'],
                'ood_fraction': ood_fraction,
                'n_hours': window_size,
                'mean_n_ood_features': window_ood_flags['n_ood_features'].mean()
            })
    
    # Remove overlapping windows (keep those with higher OOD fraction)
    if ood_windows:
        ood_windows = remove_overlapping_windows(ood_windows)
    
    print(f"✅ Found {len(ood_windows)} OOD windows")
    
    return ood_windows

def remove_overlapping_windows(windows, overlap_threshold=12):
    """Remove overlapping windows, keeping those with higher OOD fraction."""
    if len(windows) <= 1:
        return windows
    
    # Sort by OOD fraction (descending)
    windows = sorted(windows, key=lambda x: x['ood_fraction'], reverse=True)
    
    selected = []
    for window in windows:
        # Check if it overlaps with any already selected window
        overlaps = False
        for sel in selected:
            # Calculate overlap in hours
            overlap_start = max(window['start_datetime'], sel['start_datetime'])
            overlap_end = min(window['end_datetime'], sel['end_datetime'])
            
            if overlap_start < overlap_end:
                overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
                if overlap_hours >= overlap_threshold:
                    overlaps = True
                    break
        
        if not overlaps:
            selected.append(window)
    
    # Sort by start time
    selected = sorted(selected, key=lambda x: x['start_datetime'])
    
    return selected

def save_ood_windows(ood_windows, output_file):
    """Save OOD windows to CSV."""
    if not ood_windows:
        print("⚠️  No OOD windows found, not saving file.")
        return
    
    df = pd.DataFrame(ood_windows)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(ood_windows)} OOD windows to {output_file}")

def create_visualization(train_df, test_df, ood_flags, percentiles, ood_windows, output_file, region):
    """Create visualization of OOD detection."""
    print("\n📊 Creating visualization...")
    
    n_features = len(WEATHER_FEATURES)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(f'OOD Weather Detection - {region} Test Period ({TEST_START} to {TEST_END})', 
                 fontsize=16, fontweight='bold')
    
    for idx, feat in enumerate(WEATHER_FEATURES):
        ax = axes[idx]
        
        if feat not in test_df.columns or feat not in percentiles:
            ax.text(0.5, 0.5, f'Feature {feat} not available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot test data
        test_dates = pd.to_datetime(test_df['datetime_utc'])
        ax.plot(test_dates, test_df[feat], 'b-', alpha=0.6, linewidth=0.5, label='Test data')
        
        # Highlight OOD regions
        ood_mask = ood_flags[f'{feat}_ood']
        ax.scatter(test_dates[ood_mask], test_df.loc[ood_mask, feat], 
                  c='red', s=10, alpha=0.5, label='OOD points', zorder=5)
        
        # Plot percentile thresholds
        ax.axhline(percentiles[feat]['lower'], color='orange', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f'5th percentile (train)')
        ax.axhline(percentiles[feat]['upper'], color='orange', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f'95th percentile (train)')
        ax.axhline(percentiles[feat]['mean'], color='green', linestyle=':', 
                  linewidth=1, alpha=0.7, label=f'Mean (train)')
        
        # Highlight OOD windows
        for i, window in enumerate(ood_windows):
            ax.axvspan(window['start_datetime'], window['end_datetime'], 
                      alpha=0.2, color='red', zorder=1)
            if i == 0:  # Add label only once
                ax.axvspan(window['start_datetime'], window['end_datetime'], 
                          alpha=0.2, color='red', label='OOD windows', zorder=1)
        
        ax.set_ylabel(feat, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        if idx == n_features - 1:
            ax.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_file}")
    plt.close()

def generate_report(region, train_df, test_df, percentiles, ood_flags, ood_windows, output_file):
    """Generate a text report of OOD analysis."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"OOD Weather Analysis Report - {region} Test Period\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Region: {region}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Period Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Period: {TRAIN_START} to {TRAIN_END}\n")
        f.write(f"  → {len(train_df)} records ({len(train_df)/24:.1f} days)\n")
        f.write(f"Test Period: {TEST_START} to {TEST_END}\n")
        f.write(f"  → {len(test_df)} records ({len(test_df)/24:.1f} days)\n\n")
        
        f.write("Weather Features Analyzed:\n")
        f.write("-"*80 + "\n")
        for feat in WEATHER_FEATURES:
            if feat in percentiles:
                p = percentiles[feat]
                f.write(f"{feat}:\n")
                f.write(f"  5th percentile:  {p['lower']:.4f}\n")
                f.write(f"  95th percentile: {p['upper']:.4f}\n")
                f.write(f"  Training mean:   {p['mean']:.4f}\n\n")
        
        f.write("OOD Detection Summary:\n")
        f.write("-"*80 + "\n")
        total_timesteps = len(test_df)
        ood_timesteps = ood_flags['any_ood'].sum()
        ood_percentage = (ood_timesteps / total_timesteps) * 100
        
        f.write(f"Total test timesteps: {total_timesteps}\n")
        f.write(f"OOD timesteps: {ood_timesteps} ({ood_percentage:.2f}%)\n")
        f.write(f"OOD windows identified: {len(ood_windows)}\n\n")
        
        if ood_windows:
            f.write("OOD Windows Details:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'#':<4} {'Start':<20} {'End':<20} {'OOD %':<10} {'Avg OOD Feats':<15}\n")
            f.write("-"*80 + "\n")
            
            for i, window in enumerate(ood_windows, 1):
                f.write(f"{i:<4} {str(window['start_datetime']):<20} "
                       f"{str(window['end_datetime']):<20} "
                       f"{window['ood_fraction']*100:>6.1f}%   "
                       f"{window['mean_n_ood_features']:>6.2f}\n")
            
            f.write("\n")
            f.write("Summary Statistics:\n")
            f.write("-"*80 + "\n")
            ood_fractions = [w['ood_fraction'] for w in ood_windows]
            f.write(f"Average OOD fraction per window: {np.mean(ood_fractions)*100:.2f}%\n")
            f.write(f"Min OOD fraction: {np.min(ood_fractions)*100:.2f}%\n")
            f.write(f"Max OOD fraction: {np.max(ood_fractions)*100:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Identify OOD weather windows in test period')
    parser.add_argument('--region', type=str, required=True, 
                       help='Region name (e.g., Toronto, Ottawa)')
    parser.add_argument('--window_size', type=int, default=24,
                       help='Window size in hours (default: 24)')
    parser.add_argument('--ood_threshold', type=float, default=0.5,
                       help='OOD threshold (default: 0.5 = 50%%)')
    parser.add_argument('--lower_pct', type=float, default=5,
                       help='Lower percentile threshold (default: 5)')
    parser.add_argument('--upper_pct', type=float, default=95,
                       help='Upper percentile threshold (default: 95)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OOD Weather Detection - Test Period")
    print("="*80)
    print(f"Region: {args.region}")
    print(f"Training: {TRAIN_START} to {TRAIN_END}")
    print(f"Test: {TEST_START} to {TEST_END}")
    print(f"Window size: {args.window_size} hours")
    print(f"OOD threshold: {args.ood_threshold*100:.0f}%")
    print("="*80)
    
    # Load data
    df = load_data(args.region)
    
    # Split into train and test
    train_df, test_df = split_data(df)
    
    # Calculate percentiles from training data
    percentiles = calculate_percentiles(train_df, WEATHER_FEATURES, 
                                       args.lower_pct, args.upper_pct)
    
    # Identify OOD timesteps in test data
    ood_flags = identify_ood_timesteps(test_df, percentiles)
    
    print(f"\n📊 OOD Timestep Statistics:")
    print(f"   Total test timesteps: {len(test_df)}")
    print(f"   OOD timesteps: {ood_flags['any_ood'].sum()} ({ood_flags['any_ood'].mean()*100:.2f}%)")
    
    # Identify OOD windows
    ood_windows = identify_ood_windows(test_df, ood_flags, 
                                       args.window_size, args.ood_threshold)
    
    # Create output directory
    output_dir = ROOT_DIR / 'outputs' / 'ood_analysis_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save OOD windows
    windows_file = output_dir / f'ood_windows_{args.region}_test.csv'
    save_ood_windows(ood_windows, windows_file)
    
    # Create visualization
    viz_file = output_dir / f'ood_distribution_{args.region}_test.png'
    create_visualization(train_df, test_df, ood_flags, percentiles, 
                        ood_windows, viz_file, args.region)
    
    # Generate report
    report_file = output_dir / f'ood_report_{args.region}_test.txt'
    generate_report(args.region, train_df, test_df, percentiles, 
                   ood_flags, ood_windows, report_file)
    
    print("\n" + "="*80)
    print("✅ OOD Detection Complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  • OOD windows: {windows_file}")
    print(f"  • Visualization: {viz_file}")
    print(f"  • Report: {report_file}")
    print()

if __name__ == '__main__':
    main()
