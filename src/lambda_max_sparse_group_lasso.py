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
    """
    Maximal lambda

    This function calculates the maximum value for the penalty parameter lambda
    for the sparse group lasso problem.

    Parameters:
    -----------
    alpha : float
        Mixing parameter of the penalty terms. Satisfies: 0 < alpha < 1.
        The penalty term looks as follows: alpha * "lasso penalty" + (1-alpha) * "group lasso penalty".
    vector_y : np.ndarray
        Numeric vector of observations.
    vector_groups : np.ndarray
        Integer vector specifying which effect (fixed and random) belongs to which group.
    vector_weights_features : np.ndarray
        Numeric vector of weights for the vectors of fixed and random effects [b^T, u^T]^T.
    vector_beta : np.ndarray
        Numeric vector of features. At the end of this function, the random effects are
        initialized with zero, but the fixed effects are initialized via a least squares procedure.
    matrix_x : np.ndarray
        Numeric design matrix relating y to fixed and random effects [X Z].

    Returns:
    --------
    float
        The maximum value for the penalty parameter lambda.
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
