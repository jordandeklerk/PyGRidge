import numpy as np

def lambda_max_group_lasso(y, groups, feature_weights, beta, X):
    """
    Maximal lambda for group lasso regularization.

    Parameters:
    y (np.array): Numeric vector of observations.
    groups (np.array): Integer vector specifying which effect (fixed and random) belongs to which group.
    feature_weights (np.array): Numeric vector of weights for the vectors of fixed and random effects [b^T, u^T]^T.
    beta (np.array): Numeric vector of features. At the end of this function, the random effects are initialized with zero, but the fixed effects are initialized via a least squares procedure.
    X (np.array): Numeric design matrix relating y to fixed and random effects [X Z].

    Returns:
    float: The maximal lambda value.
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
            l2_norm_groups[i] = np.linalg.norm(X_transp_residual_active[group_slice]) / temp

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(np.abs(l2_norm_groups))
    return lambda_max * 1.00001