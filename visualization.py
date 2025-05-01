"""
Visualization utilities for interpolation methods.
Author: gauravvjhaa
Date: 2025-05-01 16:31:08
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def sanitize_filename(name):
    """
    Sanitize a string to make it a valid filename by removing special characters.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized string suitable for use as a filename
    """
    # Replace non-ASCII characters and special symbols
    sanitized = re.sub(r'[^\w\s.-]', '_', name)
    # Replace spaces with underscores
    sanitized = re.sub(r'[\s]+', '_', sanitized)
    return sanitized

def ensure_dir_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def visualize_interpolation(parameter, original_x, original_y, 
                            test_x, test_y, predicted_y,
                            method_name, interpolation_func, output_dir):
    """
    Create visualization of interpolation results.
    
    Args:
        parameter: Name of the parameter being interpolated
        original_x: X values of all original data points
        original_y: Y values of all original data points
        test_x: X values of test points (removed for prediction)
        test_y: Actual Y values of test points
        predicted_y: Predicted Y values at test points
        method_name: Name of interpolation method
        interpolation_func: Function that evaluates interpolation at any point
        output_dir: Directory to save visualization
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get training data (points not in test set)
    train_x = np.array([x for x in original_x if x not in test_x])
    train_y = np.array([original_y[i] for i, x in enumerate(original_x) if x not in test_x])
    
    # Plot training data points (used for interpolation)
    plt.plot(train_x, train_y, 'go', label='Training Points', markersize=5)
    
    # Generate points for smooth curve
    curve_x = np.linspace(np.min(original_x), np.max(original_x), 500)
    curve_y = np.array([interpolation_func(x) for x in curve_x])
    
    # Plot the interpolation curve
    plt.plot(curve_x, curve_y, 'g-', label=f'{method_name} Curve', linewidth=2)
    
    # Plot test points: actual values as black circles
    plt.plot(test_x, test_y, 'ko', label='Actual Values (removed)', markersize=6)
    
    # Plot predicted values as red crosses
    plt.plot(test_x, predicted_y, 'rx', label='Predicted Values', markersize=8)
    
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel(parameter)
    plt.title(f'{method_name} Interpolation for {parameter}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    safe_param = sanitize_filename(parameter)
    safe_method = sanitize_filename(method_name)
    filename = os.path.join(output_dir, f"{safe_param}_{safe_method}_interpolation.png")
    plt.savefig(filename)
    plt.close()
    
    return filename

def visualize_error_comparison(parameter, methods, errors, error_type, output_dir):
    """
    Create bar chart comparing errors across methods.
    
    Args:
        parameter: Name of the parameter
        methods: List of method names
        errors: Dictionary of error values by method
        error_type: Type of error (e.g., 'RMSE', 'MAE')
        output_dir: Directory to save visualization
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Extract error values
    error_values = [errors[method][error_type.lower()] for method in methods]
    
    # Create bar chart
    bars = plt.bar(methods, error_values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    plt.xlabel('Interpolation Method')
    plt.ylabel(error_type)
    plt.title(f'{error_type} Comparison for {parameter}')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    safe_param = sanitize_filename(parameter)
    filename = os.path.join(output_dir, f"{safe_param}_{error_type.lower()}_comparison.png")
    plt.savefig(filename)
    plt.close()
    
    return filename

def create_parameter_summary_visualization(parameter_results, metric, output_dir):
    """
    Create visualization summarizing a specific metric across all parameters.
    
    Args:
        parameter_results: Dictionary with results for each parameter
        metric: Metric to visualize (e.g., 'rmse', 'mae')
        output_dir: Directory to save visualization
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Extract methods (assuming all parameters have the same methods)
    first_param = list(parameter_results.keys())[0]
    methods = list(parameter_results[first_param].keys())
    
    # Create a dictionary to store results by method
    method_data = {method: [] for method in methods}
    parameters = []
    
    # Collect data
    for param, param_results in parameter_results.items():
        parameters.append(param)
        for method, method_results in param_results.items():
            method_data[method].append(method_results[metric])
    
    # Create the figure with wider width to accommodate parameter names
    plt.figure(figsize=(max(12, len(parameters) * 0.8), 8))
    
    # Set width of bars
    bar_width = 0.8 / len(methods)
    
    # Set positions of bars on x-axis
    r = np.arange(len(parameters))
    
    # Create bars
    for i, method in enumerate(methods):
        position = [x + bar_width * i for x in r]
        plt.bar(position, method_data[method], width=bar_width, label=method)
    
    # Add labels and title
    plt.xlabel('Parameter')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} by Method for All Parameters')
    plt.xticks([r + bar_width * (len(methods) - 1) / 2 for r in range(len(parameters))], parameters, rotation=90)
    plt.legend()
    
    # Add grid and adjust layout
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(output_dir, f"all_parameters_{metric.lower()}_comparison.png")
    plt.savefig(filename)
    plt.close()
    
    return filename