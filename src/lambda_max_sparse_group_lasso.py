"""
This module provides functionality to compute the maximum lambda value for sparse group lasso regularization.

It contains a single function, lambda_max_sparse_group_lasso, which calculates the largest
lambda value that results in a non-zero solution for the sparse group lasso problem. This is
useful for setting up a regularization path or for determining an appropriate range
of lambda values for cross-validation.
"""

import numpy as np
from seagull_bisection import seagull_bisection


def lambda_max_sparse_group_lasso(
    alpha: float,
    vector_y: np.ndarray,
    vector_groups: np.ndarray,
    vector_weights_features: np.ndarray,
    vector_beta: np.ndarray,
    matrix_x: np.ndarray,
) -> float:
    """Calculate the maximum value for the penalty parameter lambda in sparse group lasso.

    This function computes the maximum value for the penalty parameter lambda
    in the sparse group lasso problem.

    Parameters
    ----------
    alpha : float
        Mixing parameter of the penalty terms. Must satisfy 0 < alpha < 1.
        The penalty term is: alpha * "lasso penalty" + (1-alpha) * "group lasso penalty".
    vector_y : ndarray of shape (n_samples,)
        Vector of observations.
    vector_groups : ndarray of shape (n_features,)
        Integer vector specifying group membership for each effect (fixed and random).
    vector_weights_features : ndarray of shape (n_features,)
        Vector of weights for the fixed and random effects [b^T, u^T]^T.
    vector_beta : ndarray of shape (n_features,)
        Vector of features. Random effects are initialized to zero,
        fixed effects are initialized via least squares.
    matrix_x : ndarray of shape (n_samples, n_features)
        Design matrix relating y to fixed and random effects [X Z].

    Returns
    -------
    float
        The maximum value for the penalty parameter lambda.

    Notes
    -----
    The sparse group lasso penalty is a combination of the lasso and group lasso penalties:

    P(beta) = alpha * sum_j |beta_j| + (1-alpha) * sum_g ||beta_g||_2

    where beta_j are individual coefficients and beta_g are groups of coefficients.
    """
    n, p = matrix_x.shape
    number_groups = np.max(vector_groups)
    number_zeros_weights = np.sum(vector_weights_features == 0)

    vector_index_start = np.zeros(number_groups, dtype=int)
    vector_index_end = np.zeros(number_groups, dtype=int)
    vector_group_sizes = np.zeros(number_groups, dtype=int)
    vector_weights_groups = np.zeros(number_groups)

    # Create vectors of group sizes, start indices, end indices, and group weights
    for i in range(number_groups):
        group_mask = vector_groups == (i + 1)
        vector_group_sizes[i] = np.sum(group_mask)
        vector_index_start[i] = np.argmax(group_mask)
        vector_index_end[i] = vector_index_start[i] + vector_group_sizes[i] - 1
        vector_weights_groups[i] = np.sum(vector_weights_features[group_mask])

    # Treatment if unpenalized features are involved
    if number_zeros_weights > 0:
        active_mask = vector_weights_features == 0
        matrix_x_active = matrix_x[:, active_mask]
        vector_beta_active = np.linalg.solve(
            matrix_x_active.T @ matrix_x_active, matrix_x_active.T @ vector_y
        )
        vector_beta[active_mask] = vector_beta_active
        vector_residual_active = vector_y - matrix_x_active @ vector_beta_active
        vector_x_transp_residual_active = matrix_x.T @ vector_residual_active
    else:
        vector_x_transp_residual_active = matrix_x.T @ vector_y

    vector_max_groups = np.zeros(number_groups)

    # Scale and perform bisection
    for i in range(number_groups):
        if vector_weights_groups[i] == 0:
            vector_max_groups[i] = 0
        else:
            start, end = vector_index_start[i], vector_index_end[i] + 1
            vector_temp = vector_x_transp_residual_active[start:end] / (
                n * vector_weights_features[start:end]
            )
            vector_temp_absolute = np.abs(vector_temp)

            vector_temp = vector_x_transp_residual_active[start:end] / n

            left_border = 0
            right_border = np.max(vector_temp_absolute) / alpha

            vector_max_groups[i] = seagull_bisection(
                vector_group_sizes[i],
                alpha,
                left_border,
                right_border,
                vector_weights_groups[i],
                vector_weights_features[start:end],
                vector_temp,
            )

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(vector_max_groups)
    return lambda_max * 1.00001
