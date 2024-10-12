import numpy as np


def lambda_max_lasso(y, feature_weights, beta, X):
    """
    Maximal λ for LASSO regression

    Parameters:
    y (np.ndarray): numeric vector of observations.
    feature_weights (np.ndarray): numeric vector of weights for the vectors of
        fixed and random effects [b^T, u^T]^T. The entries may be permuted
        corresponding to their group assignments.
    beta (np.ndarray): numeric vector of features. At the end of this function,
        the random effects are initialized with zero, but the fixed effects are
        initialized via a least squares procedure.
    X (np.ndarray): numeric design matrix relating y to fixed and random
        effects [X Z].

    Returns:
    float: The maximal λ value for LASSO regression.
    """
    n, p = X.shape

    # Check dimensions
    if y.shape[0] != n or feature_weights.shape[0] != p or beta.shape[0] != p:
        raise ValueError("Dimension mismatch in input vectors")

    num_zeros_weights = np.sum(feature_weights == 0)

    x_transp_residual_active = np.zeros(p)

    if num_zeros_weights > 0:
        # Treatment if unpenalized features are involved
        X_active = X[:, feature_weights == 0]

        # Solve for beta_active in y = X_active * beta_active
        beta_active = np.linalg.solve(X_active.T @ X_active, X_active.T @ y)

        # Update beta with beta_active
        beta[feature_weights == 0] = beta_active

        # Calculate residual_active = y - X_active * beta_active
        residual_active = y - X_active @ beta_active

        # Calculate t(X) * residual_active
        x_transp_residual_active = X.T @ residual_active
    else:
        # Treatment if only penalized features are involved
        x_transp_residual_active = X.T @ y

    # Scale t(X)*residual_active with weight*n, if weight>0
    mask = feature_weights > 0
    x_transp_residual_active[mask] /= feature_weights[mask] * n
    x_transp_residual_active[~mask] = 0

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(np.abs(x_transp_residual_active))
    return lambda_max * 1.00001
