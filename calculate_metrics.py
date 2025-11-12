#!/usr/bin/env python3
"""
Calculate MAPE and RMSE from a CSV file with predicted_load and true_load columns.

Usage:
    python calculate_metrics.py <csv_file>
    
Example:
    python calculate_metrics.py results.csv
"""

import sys
import pandas as pd
import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    
    if not mask.any():
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE)
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE)
    
    MAE = (1/n) * Σ|y_true - y_pred|
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def calculate_r2(y_true, y_pred):
    """
    Calculate R² Score (Coefficient of Determination)
    
    R² = 1 - (SS_res / SS_tot)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    r2 = 1 - (ss_res / ss_tot)
    return r2

def main():
    if len(sys.argv) < 2:
        print("Error: CSV file path required")
        print(f"Usage: {sys.argv[0]} <csv_file>")
        print(f"Example: {sys.argv[0]} results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        # Read CSV file
        print(f"\n{'='*60}")
        print(f"Reading CSV file: {csv_file}")
        print(f"{'='*60}\n")
        
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_columns = ['predicted_load', 'true_load']
        
        # Handle different possible column names
        column_mapping = {
            'predicted_load': ['predicted_load', 'predicted', 'pred', 'y_pred', 'forecast'],
            'true_load': ['true_load', 'true', 'actual', 'y_true', 'ground_truth', 'gt']
        }
        
        # Find actual column names
        pred_col = None
        true_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if pred_col is None:
                for variant in column_mapping['predicted_load']:
                    if variant in col_lower:
                        pred_col = col
                        break
            if true_col is None:
                for variant in column_mapping['true_load']:
                    if variant in col_lower:
                        true_col = col
                        break
        
        if pred_col is None or true_col is None:
            print(f"Error: Required columns not found!")
            print(f"Available columns: {list(df.columns)}")
            print(f"\nExpected one of these patterns:")
            print(f"  Predicted: {column_mapping['predicted_load']}")
            print(f"  True/Actual: {column_mapping['true_load']}")
            sys.exit(1)
        
        print(f"Found columns:")
        print(f"  Predicted: '{pred_col}'")
        print(f"  True: '{true_col}'")
        print()
        
        # Extract values
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        
        # Check for NaN or infinite values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                       np.isinf(y_true) | np.isinf(y_pred))
        
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"Warning: Found {n_invalid} invalid values (NaN or Inf). These will be excluded.")
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
        
        # Calculate metrics
        n_samples = len(y_true)
        mape = calculate_mape(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        
        # Display results
        print(f"{'='*60}")
        print(f"METRICS RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Number of samples: {n_samples}")
        print()
        
        print(f"Primary Metrics:")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
        print(f"  RMSE (Root Mean Square Error):         {rmse:.4f}")
        print()
        
        print(f"Additional Metrics:")
        print(f"  MAE  (Mean Absolute Error):             {mae:.4f}")
        print(f"  R²   (Coefficient of Determination):    {r2:.4f}")
        print()
        
        # Summary statistics
        print(f"{'='*60}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*60}\n")
        
        print(f"True Load:")
        print(f"  Min:    {y_true.min():.2f}")
        print(f"  Max:    {y_true.max():.2f}")
        print(f"  Mean:   {y_true.mean():.2f}")
        print(f"  Median: {np.median(y_true):.2f}")
        print(f"  Std:    {y_true.std():.2f}")
        print()
        
        print(f"Predicted Load:")
        print(f"  Min:    {y_pred.min():.2f}")
        print(f"  Max:    {y_pred.max():.2f}")
        print(f"  Mean:   {y_pred.mean():.2f}")
        print(f"  Median: {np.median(y_pred):.2f}")
        print(f"  Std:    {y_pred.std():.2f}")
        print()
        
        print(f"{'='*60}\n")
        
        # Save results to a text file
        output_file = csv_file.replace('.csv', '_metrics.txt')
        with open(output_file, 'w') as f:
            f.write(f"Metrics for: {csv_file}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of samples: {n_samples}\n\n")
            f.write(f"MAPE: {mape:.4f}%\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE:  {mae:.4f}\n")
            f.write(f"R²:   {r2:.4f}\n")
        
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
