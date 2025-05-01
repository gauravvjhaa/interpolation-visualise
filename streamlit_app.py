"""
Streamlit application for visualizing interpolation methods.
Author: gauravvjhaa
Date: 2025-05-01 16:38:35
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Import functions from the interpolation module
from interpolation import (
    lagrange_interpolation, 
    newton_interpolation, 
    linear_spline, 
    cubic_spline,
    chebyshev_interpolation
)

# Define interpolation methods
INTERPOLATION_METHODS = {
    'Lagrange': lagrange_interpolation,
    'Newton': newton_interpolation,
    'Linear Spline': linear_spline,
    'Cubic Spline': cubic_spline,
    'Chebyshev': chebyshev_interpolation
}

# Page configuration
st.set_page_config(
    page_title="Interpolation Visualizer",
    page_icon="📊",
    layout="wide"
)

def get_download_link(df, filename="interpolated_data.csv"):
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Interpolated Data as CSV</a>'
    return href

def main():
    """Main function for the Streamlit app."""
    st.title("📊 Interpolation Method Visualizer")
    
    st.write("""
    ### Upload your CSV file to visualize interpolation methods
    This app will help you visualize how different interpolation methods can fill missing values in your data.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display data overview
            st.write("### Data Overview")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Show the first few rows
            st.write("#### Preview:")
            st.dataframe(df.head())
            
            # Find numeric columns with missing values
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            cols_with_missing = [col for col in numeric_cols if df[col].isna().any()]
            cols_with_complete = [col for col in numeric_cols if not df[col].isna().any()]
            
            if not numeric_cols:
                st.error("No numeric columns found in the dataset.")
                return
                
            # Display missing values info
            st.write("### Missing Values Analysis")
            
            if cols_with_missing:
                st.write("#### Columns with missing values:")
                missing_data = pd.DataFrame({
                    'Column': cols_with_missing,
                    'Missing Values': [df[col].isna().sum() for col in cols_with_missing],
                    'Percentage': [100 * df[col].isna().sum() / len(df) for col in cols_with_missing]
                })
                st.table(missing_data.style.format({'Percentage': '{:.2f}%'}))
            else:
                st.info("No columns with missing values found.")
                st.write("You can still use the app to visualize interpolation methods on complete data.")
            
            # Column selection
            st.write("### Column Selection")
            if cols_with_missing:
                st.write("Select a column with missing values to interpolate:")
                selected_col = st.selectbox("Column", cols_with_missing)
            else:
                st.write("Select a complete column for visualization:")
                selected_col = st.selectbox("Column", numeric_cols)
            
            # Method selection
            st.write("### Interpolation Method")
            selected_method = st.selectbox(
                "Choose interpolation method", 
                list(INTERPOLATION_METHODS.keys())
            )
            
            # Testing percentage (what portion of data to artificially remove)
            if selected_col in cols_with_complete:
                st.write("### Testing Configuration")
                test_percentage = st.slider(
                    "Percentage of data to artificially remove for testing", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    step=5
                )
            
            # Execute interpolation
            if st.button("Run Interpolation"):
                with st.spinner("Processing..."):
                    # Create container for results
                    result_container = st.container()
                    
                    with result_container:
                        st.write(f"### Results for {selected_col} using {selected_method}")
                        
                        # If column is complete, remove some values for testing
                        if selected_col in cols_with_complete:
                            # Get original data
                            original_x = df.index.values
                            original_y = df[selected_col].values
                            
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
                            test_y = df.loc[test_indices, selected_col].values
                            
                            # Create training dataframe (remove test points)
                            train_df = df.copy()
                            train_df.loc[test_indices, selected_col] = np.nan
                            
                            st.write(f"Removed {n_test} points ({test_percentage}%) for testing")
                            
                            # Get training data (non-NaN values)
                            train_indices = train_df.index[~train_df[selected_col].isna()].values
                            train_values = train_df.loc[train_indices, selected_col].values
                        else:
                            # If column already has missing values
                            # Get original data
                            original_x = df.index.values
                            original_y = df[selected_col].copy().values
                            
                            # Get train and test indices
                            train_indices = df.index[~df[selected_col].isna()].values
                            train_values = df.loc[train_indices, selected_col].values
                            
                            test_indices = df.index[df[selected_col].isna()].values
                            test_x = test_indices
                            
                            # We don't know the real test values in this case
                            test_y = np.array([np.nan] * len(test_indices))
                        
                        try:
                            # Create interpolation function
                            interpolator = INTERPOLATION_METHODS[selected_method](train_indices, train_values)
                            
                            # Predict test values
                            predicted_values = np.array([interpolator(x) for x in test_indices])
                            
                            # Create visualization
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot training data points (used for interpolation)
                            ax.plot(train_indices, train_values, 'go', label='Known Data Points', markersize=5)
                            
                            # Generate points for smooth curve
                            curve_x = np.linspace(np.min(original_x), np.max(original_x), 500)
                            curve_y = np.array([interpolator(x) for x in curve_x])
                            
                            # Plot the interpolation curve
                            ax.plot(curve_x, curve_y, 'g-', label=f'{selected_method} Curve', linewidth=2)
                            
                            # Plot predicted values as red crosses
                            ax.plot(test_indices, predicted_values, 'rx', label='Predicted Values', markersize=8)
                            
                            # Plot test points: actual values as black circles (if available)
                            if selected_col in cols_with_complete:
                                ax.plot(test_indices, test_y, 'ko', label='Actual Values (removed)', markersize=6)
                            
                            # Add labels and title
                            ax.set_xlabel('Index')
                            ax.set_ylabel(selected_col)
                            ax.set_title(f'{selected_method} Interpolation for {selected_col}')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                            # Create filled dataframe
                            filled_df = df.copy()
                            filled_df.loc[test_indices, selected_col] = predicted_values
                            
                            # Calculate metrics if we have actual values
                            if selected_col in cols_with_complete:
                                # Calculate absolute errors
                                absolute_errors = np.abs(test_y - predicted_values)
                                squared_errors = (test_y - predicted_values) ** 2
                                
                                # Calculate error metrics
                                mae = np.mean(absolute_errors)
                                mse = np.mean(squared_errors)
                                rmse = np.sqrt(mse)
                                
                                # Calculate R²
                                ss_total = np.sum((test_y - np.mean(test_y)) ** 2)
                                ss_residual = np.sum(squared_errors)
                                r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
                                
                                # Display metrics
                                st.write("### Interpolation Performance Metrics")
                                metrics_cols = st.columns(4)
                                metrics_cols[0].metric("MAE", f"{mae:.4f}")
                                metrics_cols[1].metric("MSE", f"{mse:.4f}")
                                metrics_cols[2].metric("RMSE", f"{rmse:.4f}")
                                metrics_cols[3].metric("R²", f"{r2:.4f}")
                            
                            # Show filled data
                            st.write("### Filled Data")
                            st.dataframe(filled_df.head(20))
                            
                            # Download link for filled data
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"interpolated_{timestamp}.csv"
                            st.markdown(get_download_link(filled_df, filename), unsafe_allow_html=True)
                            
                            # Show table with original vs predicted values
                            st.write("### Original vs. Predicted Values")
                            
                            comparison_data = pd.DataFrame({
                                'Index': test_indices,
                                'Predicted': predicted_values
                            })
                            
                            if selected_col in cols_with_complete:
                                comparison_data['Actual'] = test_y
                                comparison_data['Absolute Error'] = absolute_errors
                            
                            st.table(comparison_data)
                            
                        except Exception as e:
                            st.error(f"Error performing interpolation: {str(e)}")
                            st.exception(e)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)
    
    # Add information about interpolation methods
    with st.expander("About Interpolation Methods"):
        st.write("""
        ### Interpolation Methods
        
        1. **Lagrange Interpolation**:
           - Creates a polynomial that passes through all data points
           - Works well for small datasets but can oscillate with many points
           
        2. **Newton Interpolation**:
           - Uses divided differences to create a polynomial
           - Mathematically equivalent to Lagrange but with different formula
           
        3. **Linear Spline**:
           - Connects points with straight lines
           - Simple and stable but less smooth
           
        4. **Cubic Spline**:
           - Connects points with cubic polynomials ensuring smooth transitions
           - Produces a smooth curve with continuous first and second derivatives
           
        5. **Chebyshev Interpolation**:
           - Uses Chebyshev polynomials to reduce oscillation issues
           - More stable for high-degree polynomial interpolation
        """)

if __name__ == "__main__":
    main()