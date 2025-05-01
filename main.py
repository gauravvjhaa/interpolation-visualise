"""
Main script for interpolation methods analysis.
Author: gauravvjhaa
Date: 2025-05-01 16:31:08
"""
import os
import numpy as np
import pandas as pd
import argparse
import glob
import re
from datetime import datetime
import warnings

# Import custom modules
from interpolation import (
    lagrange_interpolation, 
    newton_interpolation, 
    linear_spline, 
    cubic_spline,
    chebyshev_interpolation
)
from visualization import visualize_interpolation, ensure_dir_exists
from evaluation import evaluate_interpolation, generate_error_report

# Suppress warnings
warnings.filterwarnings('ignore')

# Define interpolation methods
METHODS = {
    'Lagrange': lagrange_interpolation,
    'Newton': newton_interpolation,
    'Linear Spline': linear_spline,
    'Cubic Spline': cubic_spline,
    'Chebyshev': chebyshev_interpolation
}

def sanitize_filename(name):
    """
    Sanitize a string to make it a valid filename by removing special characters.
    """
    # Replace non-ASCII characters and special symbols
    sanitized = re.sub(r'[^\w\s.-]', '_', name)
    # Replace spaces with underscores
    sanitized = re.sub(r'[\s]+', '_', sanitized)
    return sanitized

def load_and_prepare_data(file_path):
    """
    Load data from CSV file and prepare for analysis.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with clean data
    """
    print(f"\nLoading data from: {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Get list of columns
        print("Available columns:")
        for col in df.columns:
            print(f"  - {col}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_valid_numeric_columns(df):
    """
    Get list of numeric columns with no missing values.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of valid column names
    """
    valid_columns = []
    
    # Check each column
    for column in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        
        # Skip columns with missing values
        if df[column].isna().any():
            print(f"Parameter {column} contains missing values, excluding...")
            continue
            
        valid_columns.append(column)
    
    print(f"\nFound {len(valid_columns)} valid numeric columns with no missing values:")
    for col in valid_columns:
        print(f"  - {col}")
    
    return valid_columns

def split_data_for_testing(df, column, test_percentage):
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame with data
        column: Column name to process
        test_percentage: Percentage of data to use for testing
        
    Returns:
        Tuple of (original_x, original_y, test_indices, test_x, test_y, train_df)
    """
    # Get original data
    original_x = df.index.values
    original_y = df[column].values
    
    # Calculate number of test points
    n_points = len(df)
    n_test = max(1, int(n_points * test_percentage / 100))
    
    # Randomly select test indices (but not the first or last point)
    np.random.seed(42)  # For reproducibility
    interior_indices = df.index[1:-1].values
    
    if len(interior_indices) <= n_test:
        # If we don't have enough interior points, use every other point
        test_indices = interior_indices[::2]
    else:
        # Otherwise randomly select from interior points
        test_indices = np.random.choice(interior_indices, size=n_test, replace=False)
    
    # Sort test indices for more intuitive visualization
    test_indices = np.sort(test_indices)
    
    # Get test values
    test_x = test_indices
    test_y = df.loc[test_indices, column].values
    
    # Create training dataframe (remove test points)
    train_df = df.copy()
    train_df.loc[test_indices, column] = np.nan
    
    print(f"Split {column} data: {n_test} test points ({test_percentage}% of {n_points} total)")
    
    return original_x, original_y, test_indices, test_x, test_y, train_df

def process_parameter(df, parameter, test_percentage, output_dir):
    """
    Process a single parameter with all interpolation methods.
    
    Args:
        df: DataFrame with data
        parameter: Column name to process
        test_percentage: Percentage of values to remove for testing
        output_dir: Directory for output files
        
    Returns:
        Dictionary with results for each method
    """
    print(f"\nProcessing parameter: {parameter}")
    
    # Split data for testing
    original_x, original_y, test_indices, test_x, test_y, train_df = \
        split_data_for_testing(df, parameter, test_percentage)
    
    # Results dictionary
    results = {}
    
    # Apply each interpolation method
    for method_name, method_func in METHODS.items():
        print(f"  Applying {method_name} interpolation...")
        
        try:
            # Get training data (non-NaN values)
            train_indices = train_df.index[~train_df[parameter].isna()].values
            train_values = train_df.loc[train_indices, parameter].values
            
            # Create interpolation function
            interpolator = method_func(train_indices, train_values)
            
            # Predict test values
            predicted_values = np.array([interpolator(x) for x in test_indices])
            
            # Evaluate results
            metrics = evaluate_interpolation(test_y, predicted_values, method_name, parameter)
            results[method_name] = metrics
            
            # Create visualization
            viz_output_dir = os.path.join(output_dir, 'visualizations')
            visualize_interpolation(
                parameter, original_x, original_y, 
                test_x, test_y, predicted_values,
                method_name, interpolator, viz_output_dir
            )
            
        except Exception as e:
            print(f"    Error with {method_name}: {e}")
            # Add empty results for failed methods
            results[method_name] = {
                'rmse': np.nan, 'mae': np.nan, 'mse': np.nan, 
                'mape': np.nan, 'r2': np.nan, 'max_error': np.nan
            }
    
    return results

def process_dataset(file_path, test_percentage, output_dir):
    """
    Process a single dataset with all valid parameters.
    
    Args:
        file_path: Path to dataset file
        test_percentage: Percentage of values to use for testing
        output_dir: Directory for output files
        
    Returns:
        Dictionary with results for each parameter
    """
    # Load the data
    df = load_and_prepare_data(file_path)
    if df is None:
        return {}
    
    # Get valid numeric columns
    valid_columns = get_valid_numeric_columns(df)
    if not valid_columns:
        print("No valid columns found in dataset.")
        return {}
    
    # Results dictionary
    results = {}
    
    # Process each parameter
    for parameter in valid_columns:
        param_results = process_parameter(df, parameter, test_percentage, output_dir)
        results[parameter] = param_results
    
    return results

def main():
    """Main function to process all datasets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interpolation Method Analysis')
    parser.add_argument('--data-dir', default='datasets', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='results', help='Directory for output files')
    parser.add_argument('--test-percent', type=float, default=20.0, help='Percentage of data to use for testing')
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    ensure_dir_exists(os.path.join(args.output_dir, 'visualizations'))
    
    # Get list of CSV files
    data_files = glob.glob(os.path.join(args.data_dir, '*.csv'))
    
    if not data_files:
        print(f"No CSV files found in {args.data_dir}")
        return
    
    print(f"Found {len(data_files)} CSV files to process")
    
    # Dictionary to store all results
    all_results = {}
    
    # Process each dataset
    for i, file_path in enumerate(sorted(data_files)):
        print(f"\nProcessing file {i+1}/{len(data_files)}: {os.path.basename(file_path)}")
        
        # Create output subdirectory for this file
        file_name = sanitize_filename(os.path.basename(file_path).split('.')[0])
        file_output_dir = os.path.join(args.output_dir, file_name)
        ensure_dir_exists(file_output_dir)
        
        # Process the dataset
        results = process_dataset(file_path, args.test_percent, file_output_dir)
        
        # Store results
        if results:
            all_results.update(results)
    
    # Generate overall error report if we have results
    if all_results:
        generate_error_report(all_results, args.output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Record start time
    start_time = datetime.now()
    print(f"Starting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Analysis completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {duration}")