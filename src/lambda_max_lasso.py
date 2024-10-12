"""
This module provides functionality to compute the maximum lambda value for LASSO regularization.

It contains a single function, lambda_max_lasso, which calculates the largest
lambda value that results in a non-zero solution for the LASSO problem. This is
useful for setting up a regularization path or for determining an appropriate range
of lambda values for cross-validation.
"""

import numpy as np


def lambda_max_lasso(y, feature_weights, beta, X):
    """
    Compute the maximal lambda for LASSO regression.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target vector.
    feature_weights : array-like of shape (n_features,)
        Weights for the feature vector [b^T, u^T]^T. The entries may be permuted
        corresponding to their group assignments.
    beta : array-like of shape (n_features,)
        Feature vector. Random effects are initialized to zero,
        fixed effects are initialized via least squares.
    X : array-like of shape (n_samples, n_features)
        Design matrix [X Z] relating y to fixed and random effects.

    Returns
    -------
    float
        The maximal lambda value for LASSO regression.

    Notes
    -----
    The maximal lambda is computed as:

    lambda_max = max_i |X_i^T r| / (n w_i)

    where X_i is the i-th column of X, r is the residual,
    n is the number of samples, and w_i is the weight for feature i.
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
