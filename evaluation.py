"""
Evaluation metrics and analysis for interpolation methods.
Author: gauravvjhaa
Date: 2025-05-01 16:31:08
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from visualization import ensure_dir_exists, visualize_error_comparison, create_parameter_summary_visualization

def calculate_metrics(true_values, predicted_values):
    """
    Calculate various error metrics between true and predicted values.
    
    Args:
        true_values: Array of actual values
        predicted_values: Array of predicted values
        
    Returns:
        Dictionary of error metrics
    """
    # Calculate absolute errors
    absolute_errors = np.abs(true_values - predicted_values)
    squared_errors = (true_values - predicted_values) ** 2
    
    # Calculate mean absolute error (MAE)
    mae = np.mean(absolute_errors)
    
    # Calculate mean squared error (MSE)
    mse = np.mean(squared_errors)
    
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate mean absolute percentage error (MAPE)
    # Avoid division by zero
    non_zero_mask = (np.abs(true_values) > 1e-10)
    if np.any(non_zero_mask):
        percentage_errors = 100 * absolute_errors[non_zero_mask] / np.abs(true_values[non_zero_mask])
        mape = np.mean(percentage_errors)
    else:
        mape = np.nan
    
    # Calculate coefficient of determination (R²)
    ss_total = np.sum((true_values - np.mean(true_values)) ** 2)
    ss_residual = np.sum(squared_errors)
    
    if ss_total > 0:
        r_squared = 1 - (ss_residual / ss_total)
    else:
        r_squared = np.nan
    
    # Calculate maximum absolute error
    max_error = np.max(absolute_errors)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r_squared,
        'max_error': max_error
    }

def evaluate_interpolation(true_values, predicted_values, method_name, parameter):
    """
    Evaluate interpolation results and return metrics.
    
    Args:
        true_values: Array of actual values
        predicted_values: Array of predicted values
        method_name: Name of the interpolation method
        parameter: Name of the parameter
        
    Returns:
        Dictionary of error metrics
    """
    # Calculate metrics
    metrics = calculate_metrics(true_values, predicted_values)
    
    # Print summary
    print(f"  {method_name} Metrics for {parameter}:")
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    MAE:  {metrics['mae']:.4f}")
    print(f"    R²:   {metrics['r2']:.4f}")
    print(f"    MAPE: {metrics['mape']:.4f}%")
    
    return metrics

def generate_error_report(results, output_dir):
    """
    Generate comprehensive error report with tables and visualizations.
    
    Args:
        results: Dictionary of results (parameter -> method -> metrics)
        output_dir: Directory to save report
    """
    # Create output directories
    report_dir = os.path.join(output_dir, 'reports')
    ensure_dir_exists(report_dir)
    
    # Create metrics tables directory
    tables_dir = os.path.join(report_dir, 'tables')
    ensure_dir_exists(tables_dir)
    
    # Create visualizations directory
    viz_dir = os.path.join(report_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    # Get list of methods from first parameter (assuming all parameters have same methods)
    first_param = next(iter(results.keys()))
    methods = list(results[first_param].keys())
    
    # Create summary dataframe for each metric
    metrics = ['rmse', 'mae', 'r2', 'mape']
    metric_dfs = {}
    
    for metric in metrics:
        # Create DataFrame with parameters as rows and methods as columns
        df = pd.DataFrame(index=results.keys(), columns=methods)
        
        # Fill the DataFrame
        for param, param_results in results.items():
            for method, method_metrics in param_results.items():
                df.loc[param, method] = method_metrics[metric]
        
        # Save to CSV
        metric_file = os.path.join(tables_dir, f"{metric}_summary.csv")
        df.to_csv(metric_file)
        metric_dfs[metric] = df
        
        # Create summary visualization across all parameters
        create_parameter_summary_visualization(results, metric, viz_dir)
    
    # Create method ranking based on average RMSE
    rmse_df = metric_dfs['rmse']
    avg_rmse = rmse_df.mean(axis=0).sort_values()
    
    # Create ranking table
    ranking_df = pd.DataFrame({
        'Average RMSE': avg_rmse,
        'Rank': range(1, len(avg_rmse) + 1)
    })
    
    ranking_file = os.path.join(tables_dir, "method_ranking.csv")
    ranking_df.to_csv(ranking_file)
    
    # Generate per-parameter visualizations
    for param, param_results in results.items():
        param_methods = list(param_results.keys())
        
        # Generate RMSE bar chart
        rmse_values = {method: metrics['rmse'] for method, metrics in param_results.items()}
        visualize_error_comparison(param, param_methods, rmse_values, 'RMSE', viz_dir)
        
        # Generate MAE bar chart
        mae_values = {method: metrics['mae'] for method, metrics in param_results.items()}
        visualize_error_comparison(param, param_methods, mae_values, 'MAE', viz_dir)
    
    # Create overall ranking visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(ranking_df.index, ranking_df['Average RMSE'], color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Interpolation Method')
    plt.ylabel('Average RMSE')
    plt.title('Overall Method Ranking (Lower RMSE is Better)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "overall_method_ranking.png"))
    plt.close()
    
    # Create comprehensive text report
    with open(os.path.join(report_dir, "interpolation_report.txt"), 'w') as f:
        f.write("Interpolation Methods Evaluation Report\n")
        f.write("=====================================\n\n")
        
        f.write("1. Method Ranking (By Average RMSE)\n")
        f.write("-----------------------------------\n")
        for i, (method, value) in enumerate(avg_rmse.items()):
            f.write(f"#{i+1}: {method} (RMSE: {value:.4f})\n")
        
        f.write("\n2. Per-Parameter Best Methods\n")
        f.write("----------------------------\n")
        
        for param in results.keys():
            param_rmse = {method: metrics['rmse'] for method, metrics in results[param].items()}
            best_method = min(param_rmse.items(), key=lambda x: x[1])[0]
            f.write(f"{param}: {best_method} (RMSE: {param_rmse[best_method]:.4f})\n")
        
        f.write("\n3. Summary of Results\n")
        f.write("--------------------\n")
        f.write("See the 'tables' directory for detailed metrics in CSV format.\n")
        f.write("See the 'visualizations' directory for comparative charts.\n")
    
    print(f"Evaluation report generated in {report_dir}")
    return report_dir