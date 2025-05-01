"""
Comprehensive interpolation methods implemented from mathematical fundamentals.
Author: gauravvjhaa
Date: 2025-05-01 16:31:08
"""
import numpy as np

def lagrange_interpolation(x_points, y_points):
    """
    Create a Lagrange interpolation polynomial.
    
    Args:
        x_points: Array of x coordinates
        y_points: Array of y coordinates
        
    Returns:
        Function that evaluates the Lagrange polynomial at any point
    """
    n = len(x_points)
    
    def lagrange_basis(i, x):
        """Calculate the i-th Lagrange basis polynomial at x."""
        result = 1.0
        for j in range(n):
            if j != i:
                result *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return result
    
    def evaluate(x):
        """Evaluate the Lagrange polynomial at point x."""
        result = 0.0
        for i in range(n):
            result += y_points[i] * lagrange_basis(i, x)
        return result
    
    return evaluate

def newton_interpolation(x_points, y_points):
    """
    Create a Newton interpolation polynomial using divided differences.
    
    Args:
        x_points: Array of x coordinates
        y_points: Array of y coordinates
        
    Returns:
        Function that evaluates the Newton polynomial at any point
    """
    n = len(x_points)
    
    # Calculate divided differences table
    coef = np.zeros([n, n])
    coef[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x_points[i+j] - x_points[i])
    
    def evaluate(x):
        """Evaluate the Newton polynomial at point x."""
        result = coef[0, 0]
        product_term = 1.0
        
        for i in range(1, n):
            product_term *= (x - x_points[i-1])
            result += coef[0, i] * product_term
            
        return result
    
    return evaluate

def linear_spline(x_points, y_points):
    """
    Create a linear spline interpolation.
    
    Args:
        x_points: Array of x coordinates (must be sorted)
        y_points: Array of y coordinates
        
    Returns:
        Function that evaluates the linear spline at any point
    """
    n = len(x_points)
    
    if n < 2:
        raise ValueError("At least two points are required for linear spline")
    
    # Ensure x_points are sorted
    if not all(x_points[i] <= x_points[i+1] for i in range(n-1)):
        idx = np.argsort(x_points)
        x_points = x_points[idx]
        y_points = y_points[idx]
    
    def evaluate(x):
        """Evaluate the linear spline at point x."""
        # Handle extrapolation cases
        if x <= x_points[0]:
            return y_points[0]
        if x >= x_points[-1]:
            return y_points[-1]
        
        # Find the appropriate segment
        for i in range(n-1):
            if x_points[i] <= x <= x_points[i+1]:
                # Linear interpolation within this segment
                t = (x - x_points[i]) / (x_points[i+1] - x_points[i])
                return (1 - t) * y_points[i] + t * y_points[i+1]
        
        # Should not reach here
        return None
    
    return evaluate

def cubic_spline(x_points, y_points):
    """
    Create a natural cubic spline interpolation.
    
    Args:
        x_points: Array of x coordinates (must be sorted)
        y_points: Array of y coordinates
        
    Returns:
        Function that evaluates the cubic spline at any point
    """
    n = len(x_points)
    
    if n < 3:
        # Fall back to linear spline if we don't have enough points
        return linear_spline(x_points, y_points)
    
    # Ensure x_points are sorted
    if not all(x_points[i] <= x_points[i+1] for i in range(n-1)):
        idx = np.argsort(x_points)
        x_points = x_points[idx]
        y_points = y_points[idx]
    
    # Step 1: Calculate the differences and steps
    h = np.diff(x_points)
    dy = np.diff(y_points)
    
    # Step 2: Set up the tridiagonal system for the second derivatives
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Natural boundary conditions: second derivatives at endpoints are zero
    A[0, 0] = 1.0
    A[n-1, n-1] = 1.0
    
    # Fill the tridiagonal matrix and the right-hand side
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (dy[i] / h[i] - dy[i-1] / h[i-1])
    
    # Step 3: Solve the system for the second derivatives
    c = np.linalg.solve(A, b)
    
    # Step 4: Calculate the remaining coefficients
    d = np.zeros(n-1)
    b_coefs = np.zeros(n-1)
    
    for i in range(n-1):
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        b_coefs[i] = dy[i] / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
    
    def evaluate(x):
        """Evaluate the cubic spline at point x."""
        # Handle extrapolation cases
        if x <= x_points[0]:
            # Linear extrapolation using the first segment
            return y_points[0] + b_coefs[0] * (x - x_points[0])
        if x >= x_points[-1]:
            # Linear extrapolation using the last segment
            return y_points[-1] + b_coefs[-1] * (x - x_points[-1])
        
        # Find the appropriate segment
        for i in range(n-1):
            if x_points[i] <= x < x_points[i+1]:
                # Calculate the polynomial value within this segment
                dx = x - x_points[i]
                return (y_points[i] + 
                        b_coefs[i] * dx + 
                        c[i] * dx**2 + 
                        d[i] * dx**3)
        
        # Should only get here if x exactly equals the last point
        return y_points[-1]
    
    return evaluate

def chebyshev_interpolation(x_points, y_points, degree=None):
    """
    Create a Chebyshev polynomial interpolation to mitigate Runge's phenomenon.
    
    Args:
        x_points: Array of x coordinates
        y_points: Array of y coordinates
        degree: Polynomial degree (default: min(15, n_points-1))
        
    Returns:
        Function that evaluates the Chebyshev interpolation at any point
    """
    n = len(x_points)
    
    if n < 2:
        raise ValueError("At least two points are required for Chebyshev interpolation")
    
    # Set default degree if not specified
    if degree is None:
        degree = min(15, n - 1)
    
    # For numerical stability, find the range of x values
    x_min, x_max = np.min(x_points), np.max(x_points)
    
    # Map original x range to [-1, 1] for Chebyshev polynomials
    x_scaled = 2 * (x_points - x_min) / (x_max - x_min) - 1
    
    # Function to evaluate Chebyshev polynomial of first kind
    def T(n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        else:
            return 2 * x * T(n-1, x) - T(n-2, x)
    
    # Calculate coefficients using least squares fitting
    A = np.zeros((n, degree + 1))
    for i in range(n):
        for j in range(degree + 1):
            A[i, j] = T(j, x_scaled[i])
    
    # Solve for coefficients
    coeffs = np.linalg.lstsq(A, y_points, rcond=None)[0]
    
    def evaluate(x):
        """Evaluate the Chebyshev interpolation at point x."""
        # Scale x to [-1, 1] range
        x_s = 2 * (x - x_min) / (x_max - x_min) - 1
        
        # Handle out-of-bounds values
        if x_s < -1:
            x_s = -1
        elif x_s > 1:
            x_s = 1
        
        # Evaluate the Chebyshev expansion
        result = 0.0
        for j in range(degree + 1):
            result += coeffs[j] * T(j, x_s)
            
        return result
    
    return evaluate