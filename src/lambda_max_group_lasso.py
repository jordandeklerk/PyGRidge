"""
This module provides functionality to compute the maximum lambda value for group
lasso regularization.

It contains a single function, lambda_max_group_lasso, which calculates the
largest lambda value that results in a non-zero solution for the group lasso
problem. This is useful for setting up a regularization path or for determining
an appropriate range of lambda values for cross-validation.
"""

import numpy as np


def lambda_max_group_lasso(y, groups, feature_weights, beta, X):
    """Compute the maximal lambda for group lasso regularization.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target vector.
    groups : array-like of shape (n_features,)
        Integer vector specifying group membership for each feature.
    feature_weights : array-like of shape (n_features,)
        Weights for the feature vector [b^T, u^T]^T.
    beta : array-like of shape (n_features,)
        Feature vector. Random effects are initialized to zero,
        fixed effects are initialized via least squares.
    X : array-like of shape (n_samples, n_features)
        Design matrix [X Z] relating y to fixed and random effects.

    Returns
    -------
    float
        The maximal lambda value.

    Notes
    -----
    Computes the maximal lambda value for group lasso regularization:
    1. Initialize group-specific variables.
    2. Handle unpenalized features if present.
    3. Scale X.T * residuals in groups.
    4. Determine lambda_max and apply numeric correction.

    The formula for lambda_max is:

    lambda_max = max_i (||X_i.T r||_2) / (n w_i sqrt(|G_i|))

    where X_i is the design matrix for group i, r is the residual,
    n is the number of samples, w_i is the weight for group i,
    and |G_i| is the size of group i.
    """
    n, p = X.shape
    num_groups = np.max(groups)
    lambda_max = 0.0
    num_zeros_weights = np.sum(feature_weights == 0.0)

    index_start = np.zeros(num_groups, dtype=int)
    index_end = np.zeros(num_groups, dtype=int)
    group_weights = np.zeros(num_groups)
    l2_norm_groups = np.zeros(num_groups)

    # Create vectors of group sizes, start indices, end indices, and group weights
    for i in range(num_groups):
        group_mask = groups == (i + 1)
        group_indices = np.where(group_mask)[0]
        index_start[i] = group_indices[0]
        index_end[i] = group_indices[-1]
        group_weights[i] = np.sum(feature_weights[group_mask])

    group_weights = np.sqrt(group_weights)

    # Treatment if unpenalized features are involved
    if num_zeros_weights > 0:
        active_mask = feature_weights == 0.0
        X_active = X[:, active_mask]
        beta_active = np.linalg.solve(X_active.T @ X_active, X_active.T @ y)
        beta[active_mask] = beta_active
        residual_active = y - X_active @ beta_active
        X_transp_residual_active = X.T @ residual_active
    else:
        X_transp_residual_active = X.T @ y

    # Scale t(X)*res_A in groups with n*weight*sqrt(group_size), if weight>0
    for i in range(num_groups):
        if group_weights[i] == 0.0:
            l2_norm_groups[i] = 0.0
        else:
            temp = n * group_weights[i]
            group_slice = slice(index_start[i], index_end[i] + 1)
            l2_norm_groups[i] = (
                np.linalg.norm(X_transp_residual_active[group_slice]) / temp
            )

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(np.abs(l2_norm_groups))
    return lambda_max * 1.00001
