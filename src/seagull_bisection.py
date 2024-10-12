"""
This module implements a bisection algorithm for finding the smallest positive root of a polynomial.

The main function, seagull_bisection, is used in sparse group lasso optimization to find
the root of a specific polynomial arising from the proximal operator calculation.
"""

import numpy as np


def sqrt_double(x):
    return np.sqrt(x)


def seagull_bisection(
    rows: int,
    alpha: float,
    left_border: float,
    right_border: float,
    group_weight: float,
    vector_weights: np.ndarray,
    vector_in: np.ndarray,
) -> float:
    """
    Internal bisection algorithm for finding the smallest positive root of a polynomial.

    This algorithm finds the smallest positive root of a polynomial
    of second degree in lambda. Bisection is an implicit algorithm, i.e.,
    it calls itself until a certain precision is reached.

    Parameters
    ----------
    rows : int
        The length of the input vectors.
    alpha : float
        Mixing parameter of the penalty terms. Must satisfy 0 < alpha < 1.
        The penalty term is: alpha * "lasso penalty" + (1-alpha) * "group lasso penalty".
    left_border : float
        Value of the left border of the current interval that for sure harbors a root.
    right_border : float
        Value of the right border of the current interval that for sure harbors a root.
    group_weight : float
        A multiplicative scalar which is part of the polynomial.
    vector_weights : ndarray of shape (rows,)
        An input vector of multiplicative scalars which are part of the polynomial.
        This vector is a subset of the vector of weights for features.
    vector_in : ndarray of shape (rows,)
        Another input vector which is required to compute the value of the polynomial.

    Returns
    -------
    float
        The smallest positive root of the polynomial, or the center point
        of the interval containing the root if a certain precision is reached.

    Raises
    ------
    ValueError
        If alpha is not between 0 and 1 (exclusive), if left_border is greater than
        or equal to right_border, or if the lengths of vector_weights and vector_in
        do not match the specified number of rows.
    RuntimeError
        If the bisection algorithm does not converge after the maximum number of iterations.

    Notes
    -----
    The algorithm uses a bisection method to find the root of the polynomial:

    f(lambda) = sum_i max(0, |x_i| - alpha * lambda * w_i)^2 - (1-alpha)^2 * lambda^2 * g

    where x_i are the elements of vector_in, w_i are the elements of vector_weights,
    and g is the group_weight.
    """
    # Input validation
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1 (exclusive)")
    if left_border >= right_border:
        raise ValueError("Left border must be less than right border")
    if rows != len(vector_weights) or rows != len(vector_in):
        raise ValueError("Rows must match the length of vector_weights and vector_in")

    max_value = max(abs(left_border), abs(right_border), np.max(np.abs(vector_in)))
    tolerance = max(1e-13, 1e-13 * max_value)  # Adjust tolerance for large values
    max_iterations = 10000  # Increase maximum iterations

    for _ in range(max_iterations):
        mid_point = 0.5 * (left_border + right_border)
        func_left = 0.0
        func_mid = 0.0
        func_right = 0.0

        # Calculate the value of the function at the left border, the mid point, and the right border of the interval
        for i in range(rows):
            if vector_in[i] < 0.0:
                temp_left = -vector_in[i] - alpha * left_border * vector_weights[i]
                temp_mid = -vector_in[i] - alpha * mid_point * vector_weights[i]
                temp_right = -vector_in[i] - alpha * right_border * vector_weights[i]
            else:
                temp_left = vector_in[i] - alpha * left_border * vector_weights[i]
                temp_mid = vector_in[i] - alpha * mid_point * vector_weights[i]
                temp_right = vector_in[i] - alpha * right_border * vector_weights[i]

            func_left += max(0, temp_left) ** 2
            func_mid += max(0, temp_mid) ** 2
            func_right += max(0, temp_right) ** 2

        func_left -= (1.0 - alpha) ** 2 * left_border**2 * group_weight
        func_mid -= (1.0 - alpha) ** 2 * mid_point**2 * group_weight
        func_right -= (1.0 - alpha) ** 2 * right_border**2 * group_weight

        # Check for change of sign within sub-intervals and redo bisection
        if func_left * func_mid <= 0.0:
            if abs(left_border - mid_point) <= tolerance:
                return mid_point
            right_border = mid_point
        elif func_mid * func_right <= 0.0:
            if abs(mid_point - right_border) <= tolerance:
                return mid_point
            left_border = mid_point
        else:
            # If no sign change, return the point with the smallest absolute function value
            abs_func_left = abs(func_left)
            abs_func_mid = abs(func_mid)
            abs_func_right = abs(func_right)
            min_abs_func = min(abs_func_left, abs_func_mid, abs_func_right)
            if min_abs_func == abs_func_left:
                return left_border
            elif min_abs_func == abs_func_mid:
                return mid_point
            else:
                return right_border

    raise RuntimeError(f"Bisection did not converge after {max_iterations} iterations")
