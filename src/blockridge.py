"""
Blockridge Module
=================

This module implements a comprehensive suite of Ridge regression predictors and related utilities,
designed for efficient and flexible regularized linear regression, particularly suitable for
high-dimensional data where the number of features p may be comparable to or exceed the number
of samples n.

Overview
--------

Ridge regression addresses the problem of multicollinearity in linear regression models and
provides a way to regularize the estimation of regression coefficients in high-dimensional
settings. It adds a penalty term to the least squares objective function to shrink the
coefficients towards zero, thereby reducing variance at the cost of introducing some bias.

Mathematical Formulation
------------------------

Given a dataset with response vector Y ∈ R^n and design matrix X ∈ R^(n×p), Ridge regression
solves the following optimization problem:

minimize over β ∈ R^p:

    L(β) = || Y - Xβ ||_2^2 + λ || β ||_2^2

where:

- β is the vector of regression coefficients.
- λ ≥ 0 is the regularization parameter controlling the amount of shrinkage.

The closed-form solution to this problem is:

    β̂ = (X^T X + λ I_p)⁻¹ X^T Y

where I_p is the p-dimensional identity matrix.

Group-wise Regularization
-------------------------

In many applications, features are naturally grouped (e.g., genes in biological pathways, pixels
in image patches). Group-wise Ridge regression allows for different regularization parameters λ_g
for each group g of features, leading to the optimization problem:

minimize over β ∈ R^p:

    L(β) = || Y - Xβ ||_2^2 + ∑_(g=1)^G λ_g || β_g ||_2^2

where:

- β_g is the subvector of β corresponding to group g.
- G is the total number of groups.

This approach allows for differential penalization of feature groups, which can improve model
performance when some groups are believed to be more informative than others.

Computational Challenges
------------------------

Computing the Ridge regression solution directly via matrix inversion can be computationally
expensive or numerically unstable, especially when dealing with high-dimensional data (large p)
or when p > n.

This module addresses these challenges by implementing efficient computational methods leveraging
matrix identities and decompositions, such as:

- Cholesky decomposition for p < n
- Woodbury matrix identity for p > n
- Sherman-Morrison formula for very large p

Regularization Parameter Tuning
-------------------------------

Choosing appropriate values for the regularization parameters λ_g is crucial for model
performance. This module provides advanced methods for tuning λ_g, including:

- Moment-based approaches that use statistical properties of the data.
- Regularization path analysis, which studies how the solution varies with λ_g.

Key Components
--------------

1. Abstract and Concrete Ridge Predictors

   - AbstractRidgePredictor: Defines a common interface for all Ridge predictors in the module,
     enforcing consistency and facilitating interchangeability.
   - CholeskyRidgePredictor: Utilizes Cholesky decomposition, suitable when p < n. It efficiently
     computes the solution by exploiting the positive-definite nature of X^T X + Λ.
   - WoodburyRidgePredictor: Employs the Woodbury matrix identity, advantageous when p > n. It
     inverts the regularized covariance matrix by converting it into a problem involving
     inversion of an n x n matrix.
   - ShermanMorrisonRidgePredictor: Applies the Sherman-Morrison formula for rank-one updates,
     efficient for very high-dimensional data where p >> n.

2. BasicGroupRidgeWorkspace

   - Manages the entire Ridge regression workflow.
   - Handles model fitting, prediction, and updating regularization parameters.
   - Computes leverage scores and error metrics such as the leave-one-out (LOO) error.
   - Leverages the appropriate Ridge predictor based on the dimensionality of the data.

3. Regularization Parameter Tuning

   - lambda_lolas_rule: Compute the regularization parameter λ using the Panagiotis Lolas rule.
     The Lolas rule provides a heuristic for selecting the regularization parameter based on
     the model's degrees of freedom and the trace of X^T X. This method balances the complexity
     of the model against its fit to the training data.
   - MomentTunerSetup: Prepares moment-based statistics (e.g., norms of coefficient vectors,
     design matrix properties) required for tuning λ_g.
   - sigma_squared_path: Computes the regularization path by varying the noise variance σ²,
     allowing analysis of how the solution and error metrics change with different levels of
     assumed noise.
   - get_lambdas and get_alpha_s_squared: Helper functions that compute λ_g based on σ² and
     moment-based statistics.

Detailed Mathematical Explanations
----------------------------------

Cholesky Decomposition in Ridge Regression:

When p < n, the matrix X^T X + Λ is p x p and positive definite (assuming λ > 0), making it
suitable for Cholesky decomposition:

    X^T X + Λ = L L^T

where L is a lower triangular matrix. This decomposition allows efficient solving of the linear
system for β̂ without explicitly computing the inverse.

Woodbury Matrix Identity:

When p > n, inverting the p x p matrix X^T X + Λ is computationally expensive. The Woodbury
identity allows us to express the inverse in terms of an n x n matrix:

    (X^T X + Λ)⁻¹ = Λ⁻¹ - Λ⁻¹ X^T (I_n + X Λ⁻¹ X^T)⁻¹ X Λ⁻¹

This reduces computational cost since inverting an n x n matrix is more efficient when n << p.

Sherman-Morrison Formula:

For very large p, even the Woodbury identity may be computationally intensive. The
Sherman-Morrison formula is used for rank-one updates to the inverse of X^T X + Λ, which can be
efficient when sequentially updating the regularization parameters or when the data matrix X has
special structure.

Moment-based Regularization Parameter Tuning:

The module includes methods for computing optimal λ_g based on moments of the data and
coefficients:

- α_g^2: The squared norm of the coefficients for group g, adjusted for σ^2.
- γ_g: The ratio of the number of features in group g to the total number of samples.
- λ_g: Computed as λ_g = (σ^2 γ_g) / α_g^2

This approach ensures that groups with larger coefficients or more features are appropriately
regularized.

Regularization Path Analysis

By computing the solution for a range of σ^2 values, users can analyze how the coefficients and
error metrics change, aiding in the selection of optimal λ_g and understanding the model's
sensitivity to regularization.
"""


from abc import ABC, abstractmethod
from typing import List, Union, TypeVar
import numpy as np
from scipy.linalg import cho_solve
from ..src.groupedfeatures import GroupedFeatures
from ..src.nnls import nonneg_lsq, NNLSError
import warnings

T = TypeVar("T")


class RidgeRegressionError(Exception):
    """Base exception class for Ridge regression errors."""

    pass


class InvalidDimensionsError(RidgeRegressionError):
    """Exception raised when matrix dimensions are incompatible."""

    pass


class SingularMatrixError(RidgeRegressionError):
    """Exception raised when a matrix is singular or nearly singular."""

    pass


class NumericalInstabilityError(RidgeRegressionError):
    """Exception raised when numerical instability is detected."""

    pass


class AbstractRidgePredictor(ABC):
    """
    Abstract base class for Ridge regression predictors.

    This class defines the interface that all concrete Ridge predictors must implement,
    ensuring consistency in how regularization parameters are updated, and how various
    mathematical operations related to Ridge regression are performed.

    Ridge Regression seeks to minimize the following objective function:

    min_β { ||Y - Xβ||_2^2 + λ||β||_2^2 }

    Where:
    - Y is the response vector with dimensions (n, ).
    - X is the design matrix with dimensions (n, p).
    - β is the coefficient vector with dimensions (p, ).
    - λ is the regularization parameter.

    The solution to this optimization problem is given by:

    β = (X^T X + λI_p)⁻¹ X^T Y

    This abstract class outlines methods for updating regularization parameters, computing
    traces, performing matrix inversions, and solving linear systems essential for Ridge regression.
    """

    @abstractmethod
    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        """
        Update the regularization parameters (λ) for the predictor based on feature groups.

        Args:
            groups (GroupedFeatures): The grouped features object defining feature groupings.
            lambdas (np.ndarray): The new λ values for each group.
        """
        pass

    @abstractmethod
    def trace_XtX(self) -> float:
        """
        Compute the trace of X^T X.

        Returns:
            float: The trace of X^T X.
        """
        pass

    @abstractmethod
    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        """
        Compute (X^T X + Λ)⁻¹ X^T X.

        Returns:
            np.ndarray: The result of the computation.
        """
        pass

    @abstractmethod
    def ldiv(self, B: np.ndarray) -> np.ndarray:
        """
        Solve the linear system (X^T X + Λ) x = B.

        Args:
            B (np.ndarray): The right-hand side matrix or vector.

        Returns:
            np.ndarray: The solution vector x.
        """
        pass


class CholeskyRidgePredictor(AbstractRidgePredictor):
    """
    Ridge predictor using Cholesky decomposition for efficient matrix inversion.

    Suitable for scenarios where the number of features p is less than the number
    of samples n, leveraging the properties of Cholesky decomposition to solve
    the Ridge regression problem efficiently.

    1. Design Matrix and Regularization Matrix:

       - Design Matrix (X):

         X is an n × p matrix.

       - Regularization Matrix (Λ):

         Λ is a diagonal matrix with λ₁, λ₂, ..., λₚ on the diagonal.

    2. Calculating X^T X:

       Compute the matrix multiplication of X transposed and X.

    3. Augmented Matrix (X^T X + Λ):

       Add the regularization matrix Λ to X^T X.

    4. Cholesky Decomposition:

       Decompose (X^T X + Λ) into L × L^T, where L is a lower triangular matrix.

    5. Solving Linear Systems:

       - Compute (X^T X + Λ)⁻¹ X^T X using the Cholesky factors.
       - Solve (X^T X + Λ) x = B for x using forward and backward substitution based on the Cholesky factors.
    """

    def __init__(self, X: np.ndarray):
        """
        Initialize the Cholesky Ridge predictor.

        Args:
            X (np.ndarray): The design matrix.

        Raises:
            InvalidDimensionsError: If X is not a 2D array.
            ValueError: If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n, self.p = X.shape
        self.XtX = np.dot(X.T, X) / self.n
        self.XtXp_lambda = self.XtX + np.eye(self.p)  # Initialize with identity matrix
        self.update_cholesky()
        self.lower = True

    def update_cholesky(self):
        """
        Update the Cholesky decomposition of (X^T X + Λ).

        This method computes the Cholesky decomposition.

        Raises:
            SingularMatrixError: If the matrix is not positive definite.
        """
        try:
            self.XtXp_lambda_chol = np.linalg.cholesky(self.XtXp_lambda)
        except np.linalg.LinAlgError:
            raise SingularMatrixError(
                "Failed to compute Cholesky decomposition. Matrix may not be positive"
                " definite."
            )

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        """
        Update the regularization parameters and recompute the Cholesky decomposition.

        Args:
            groups (GroupedFeatures): The grouped features object.
            lambdas (np.ndarray): The new λ values for each group.
        """
        diag = groups.group_expand(lambdas)
        self.XtXp_lambda = self.XtX + np.diag(diag)
        self.update_cholesky()

    def trace_XtX(self) -> float:
        """
        Compute the trace of X^T X.

        Returns:
            float: The trace of X^T X.
        """
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        """
        Compute (X^T X + Λ)⁻¹ X^T X using Cholesky decomposition.

        Returns:
            np.ndarray: The result of the computation.
        """
        return cho_solve((self.XtXp_lambda_chol, self.lower), self.XtX)

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        """
        Solve the system (X^T X + Λ) x = B using Cholesky decomposition.

        Args:
            B (np.ndarray): The right-hand side of the equation.

        Returns:
            np.ndarray: The solution vector x.
        """
        return cho_solve((self.XtXp_lambda_chol, self.lower), B)


class WoodburyRidgePredictor(AbstractRidgePredictor):
    """
    Ridge predictor using the Woodbury matrix identity for efficient matrix inversion.

    This class is suitable for scenarios where the number of features p is greater than
    the number of samples n, leveraging the Woodbury matrix identity to solve
    the Ridge regression problem efficiently.

    The Woodbury matrix identity states:
    (A + UCV)^(-1) = A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)

    In the context of Ridge regression:
    A = Λ (diagonal matrix of regularization parameters)
    U = X^T
    C = I (identity matrix)
    V = X
    """

    def __init__(self, X: np.ndarray):
        """
        Initialize the Woodbury Ridge predictor.

        Args:
            X (np.ndarray): The design matrix.

        Raises:
            InvalidDimensionsError: If X is not a 2D array.
            ValueError: If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n, self.p = X.shape
        self.X = X
        self.XtX = X.T @ X
        self.A_inv = np.eye(self.p)
        self.U = X.T
        self.V = X

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        """
        Update the regularization parameters and recompute the inverse matrix.

        Args:
            groups (GroupedFeatures): The grouped features object.
            lambdas (np.ndarray): The new λ values for each group.

        Raises:
            ValueError: If lambdas contain non-positive values.
        """
        if np.any(lambdas <= 0):
            raise ValueError("Lambda values must be positive.")

        diag = np.array(groups.group_expand(lambdas))
        self.A_inv = np.diag(1 / diag)
        self.woodbury_update()

    def woodbury_update(self):
        """
        Apply the Woodbury matrix identity to update the inverse matrix.

        Raises:
            NumericalInstabilityError: If numerical instability is detected during the update.
        """
        try:
            eye = np.eye(self.n)
            AU = self.A_inv @ self.U
            inv_term = np.linalg.inv(eye + self.V @ AU)
            self.A_inv -= AU @ inv_term @ self.V @ self.A_inv
        except np.linalg.LinAlgError:
            raise NumericalInstabilityError(
                "Numerical instability detected in Woodbury update."
            )

    def trace_XtX(self) -> float:
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        return self.A_inv @ self.XtX

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        return self.A_inv @ B


class ShermanMorrisonRidgePredictor(AbstractRidgePredictor):
    """
    Ridge predictor using the Sherman-Morrison formula for efficient matrix updates.

    This class is suitable for scenarios where the number of features p is much greater
    than the number of samples n, leveraging the Sherman-Morrison formula to
    efficiently update the inverse of (X^T X + Λ) as Λ changes.

    The Sherman-Morrison formula states:
    (A + uv^T)^(-1) = A^(-1) - (A^(-1)u v^T A^(-1)) / (1 + v^T A^(-1) u)

    Where A is the current inverse, and uv^T represents a rank-one update.
    """

    def __init__(self, X: np.ndarray):
        """
        Initialize the Sherman-Morrison Ridge predictor.

        Args:
            X (np.ndarray): The design matrix.

        Raises:
            InvalidDimensionsError: If X is not a 2D array.
            ValueError: If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n, self.p = X.shape
        self.X = X
        self.XtX = self.X.T @ self.X / self.n
        self.A = np.eye(self.p) + self.XtX  # A = I + XtX
        self.A_inv = np.linalg.inv(self.A)  # Initialize A_inv as inv(A)
        self.U = self.X.T / np.sqrt(self.n)  # Precompute for efficiency
        self.V = self.X / np.sqrt(self.n)

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        """
        Update the regularization parameters (lambda) and adjust A_inv accordingly.

        Args:
            groups (GroupedFeatures): The grouped features object.
            lambdas (np.ndarray): The new λ values for each group.

        Raises:
            ValueError: If lambdas contain negative values.
        """
        if np.any(lambdas < 0):
            raise ValueError("Lambda values must be non-negative.")

        diag = groups.group_expand(lambdas)
        self.A = np.diag(diag) + self.XtX  # Update A with new Lambda
        try:
            self.A_inv = np.linalg.inv(self.A)  # Recompute A_inv
        except np.linalg.LinAlgError:
            raise SingularMatrixError("Failed to invert A. Matrix may be singular.")

    def sherman_morrison(self, u: np.ndarray, v: np.ndarray):
        """
        Apply the Sherman-Morrison formula to update self.A_inv with rank-one update u v^T.

        Args:
            u (np.ndarray): Left vector for rank-one update.
            v (np.ndarray): Right vector for rank-one update.

        Raises:
            NumericalInstabilityError: If numerical instability is detected during the update.
        """
        Au = self.A_inv @ u
        vA = v @ self.A_inv
        denominator = 1.0 + v @ Au
        if abs(denominator) < 1e-10:
            raise NumericalInstabilityError(
                "Denominator in Sherman-Morrison update is close to zero."
            )
        self.A_inv -= np.outer(Au, vA) / denominator

    def trace_XtX(self) -> float:
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        return self.ldiv(self.XtX)

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        """
        Solve (XtX + Lambda)^-1 * B using the precomputed A_inv.
        """
        if B.ndim == 1:
            return self.A_inv @ B
        else:
            return self.A_inv @ B

    @staticmethod
    def sherman_morrison_formula(
        A: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """
        Applies the Sherman-Morrison formula to matrix A with vectors u and v.
        Returns the updated matrix.
        """
        Au = A @ u
        vA = v @ A
        denominator = 1.0 + v @ Au
        if denominator == 0:
            raise ValueError("Denominator in Sherman-Morrison update is zero.")
        return A - np.outer(Au, vA) / denominator


class BasicGroupRidgeWorkspace:
    """
    A workspace for performing group Ridge regression, including fitting and prediction.

    This class manages the entire workflow of group-wise Ridge regression, handling
    the fitting process, parameter updates, predictions, and evaluation metrics. It
    leverages Ridge predictors (Cholesky, Woodbury, Sherman-Morrison) based on the dimensionality of
    the data to ensure computational efficiency.

    1. Ridge Regression Initialization:

       - Design Matrix (X) and Response Vector (Y):

         X is an n by p matrix, Y is a vector of length n.

       - Grouped Features (G):

         G defines feature groups for applying group-wise regularization.

       - Initial Ridge Predictor:

         Uses CholeskyRidgePredictor, WoodburyRidgePredictor, or ShermanMorrisonRidgePredictor
         to compute:

         β = (X^T X + Λ)⁻¹ X^T Y

    2. Regularization Parameters (λ):

       Initialized as a vector of ones corresponding to each feature group:

       λ is a vector of length g, where g is the number of groups.

    3. Updating Regularization Parameters:

       - Expand Group-wise λ to Feature-wise Λ:

         Λ is a diagonal matrix where each diagonal element corresponds to a feature's λ value, expanded from group-wise λs.

       - Coefficient Updates:

         β = (X^T X + Λ)⁻¹ X^T Y

    4. Predicted Values (Ŷ):

       Ŷ = X * β

    5. Leverage Scores:

       hᵢ = Xᵢ * (X^T X + Λ)⁻¹ * X^T

    6. Leave-One-Out (LOO) Error:

       LOO Error = (1/n) * Σ_{i=1}^{n} [ (Yᵢ - Ŷᵢ)^2 / (1 - hᵢ)^2 ]

    7. Mean Squared Error (MSE) on Test Data:

       MSE = (1/m) * Σ_{j=1}^{m} (Y_testⱼ - X_testⱼ * β)²
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, groups: GroupedFeatures):
        """
        Initialize the BasicGroupRidgeWorkspace.

        Args:
            X (np.ndarray): The design matrix.
            Y (np.ndarray): The target vector.
            groups (GroupedFeatures): The grouped features object.
        """
        if not groups.ps:
            raise ValueError("GroupedFeatures must contain at least one group.")
        if any(p == 0 for p in groups.ps):
            raise ValueError("GroupedFeatures groups must have non-zero sizes.")
        self.X = X
        self.Y = Y
        self.groups = groups
        self.n, self.p = X.shape

        # Initialize predictor based on p and n
        if self.p <= self.n:
            self.predictor = CholeskyRidgePredictor(X)
        elif self.p > self.n and self.p < 4 * self.n:
            self.predictor = WoodburyRidgePredictor(X)
        else:
            self.predictor = ShermanMorrisonRidgePredictor(X)

        self.XtY = np.dot(X.T, Y) / self.n
        self.lambdas = np.ones(groups.num_groups)
        self.update_lambda_s(self.lambdas)
        self.beta_current = self.predictor.ldiv(self.XtY)
        self.Y_hat = np.dot(X, self.beta_current)

        # Compute (X^T X + Λ)^{-1}
        self.XtXp_lambda_inv = self.predictor.ldiv(np.eye(self.p))

        self.moment_setup = MomentTunerSetup(self)

    def update_lambda_s(self, lambdas: np.ndarray):
        """
        Update the regularization parameters (λ).

        Args:
            lambdas (np.ndarray): New λ values for each group.
        """
        self.lambdas = lambdas
        self.predictor.update_lambda_s(self.groups, self.lambdas)
        self.XtXp_lambda_div_Xt = self.predictor.ldiv(self.X.T)

    def ngroups(self) -> int:
        """
        Return the number of feature groups.

        Returns:
            int: Number of groups.
        """
        return self.groups.ngroups()

    def coef(self) -> np.ndarray:
        """
        Return the current coefficient estimates (β).

        Returns:
            np.ndarray: Coefficient vector β.
        """
        return self.beta_current

    def islinear(self) -> bool:
        """
        Check if the model is linear.

        Returns:
            bool: Always returns True for Ridge regression.
        """
        return True

    def leverage(self) -> np.ndarray:
        """
        Return the leverage scores (hᵢ).

        Returns:
            np.ndarray: Leverage scores vector.
        """
        return self.leverage_store

    def modelmatrix(self) -> np.ndarray:
        """
        Return the design matrix (X).

        Returns:
            np.ndarray: Design matrix X.
        """
        return self.X

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data.

        Args:
            X_new (np.ndarray): New design matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        return np.dot(X_new, self.coef())

    def response(self) -> np.ndarray:
        """
        Return the response variable (Y).

        Returns:
            np.ndarray: Response vector Y.
        """
        return self.Y

    def loo_error(self) -> float:
        """
        Compute the leave-one-out (LOO) error.

        Returns:
            float: The computed LOO error.
        """
        return np.mean((self.Y - self.Y_hat) ** 2 / (1.0 - self.leverage_store) ** 2)

    def mse_ridge(self, X_test: np.ndarray, Y_test: np.ndarray) -> float:
        """
        Compute the mean squared error (MSE) on test data.

        Args:
            X_test (np.ndarray): Test design matrix.
            Y_test (np.ndarray): Test response vector.

        Returns:
            float: The computed MSE.
        """
        return np.mean((Y_test - np.dot(X_test, self.beta_current)) ** 2)

    def fit(self, lambdas: Union[np.ndarray, dict]):
        """
        Fit the Ridge regression model with given regularization parameters.

        Args:
            lambdas (Union[np.ndarray, dict]): The regularization parameters.
                Can be a numpy array or a dictionary mapping group names to λ values.

        Returns:
            float: The leave-one-out (LOO) error after fitting.
        """
        if isinstance(lambdas, dict):
            lambdas = np.array(list(lambdas.values()))
        self.lambdas = lambdas
        self.predictor.update_lambda_s(self.groups, self.lambdas)
        self.beta_current = self.predictor.ldiv(self.XtY)
        self.Y_hat = np.dot(self.X, self.beta_current)
        self.XtXp_lambda_div_Xt = self.predictor.ldiv(self.X.T).T / self.n
        self.leverage_store = np.sum(self.X * self.XtXp_lambda_div_Xt, axis=1)
        return self.loo_error()


def lambda_lolas_rule(rdg: BasicGroupRidgeWorkspace, multiplier: float = 0.1) -> float:
    """
    Compute the regularization parameter λ using the Panagiotis Lolas rule.

    The Lolas rule provides a heuristic for selecting the regularization parameter based on
    the model's degrees of freedom and the trace of X^T X. This method balances
    the complexity of the model against its fit to the training data.

    The λ is computed as:

    λ = multiplier * (p^2 / n) / trace(X^T X)

    Where:
    - multiplier is a scalar factor (default: 0.1).
    - p is the number of features.
    - n is the number of samples.
    - trace(X^T X) is the trace of the covariance matrix X^T X.

    This formula scales the regularization parameter based on both the dimensionality
    of the data and the trace of the covariance matrix, ensuring that λ is
    appropriately tuned for the given dataset.

    Args:
        rdg (BasicGroupRidgeWorkspace): The Ridge regression workspace containing model parameters.
        multiplier (float, optional): A scalar multiplier for the rule. Defaults to 0.1.

    Returns:
        float: The computed λ value.

    Raises:
        ValueError: If multiplier is not positive or if trace(X^T X) is zero.
    """
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")

    trace_XtX = rdg.predictor.trace_XtX()
    if trace_XtX == 0:
        raise ValueError("Trace of X^T X is zero, leading to division by zero.")

    return multiplier * rdg.p**2 / rdg.n / trace_XtX


class MomentTunerSetup:
    """
    Setup for the moment-based tuning of regularization parameters.

    This class prepares and computes moment-based statistics required for tuning the
    regularization parameters (λ) in Ridge regression. By leveraging moments
    of the coefficients and the design matrix, it facilitates principled selection of
    λ values that balance bias and variance.

    1. Initialization Parameters:

       - Workspace (rdg):

         An instance of BasicGroupRidgeWorkspace containing the model's current state.

    2. Computing Moments:

       - Beta Norms Squared (||βᵍ||^2):

         For each group g:

         ||βᵍ||^2 = Σ_{j ∈ g} |βⱼ|^2

       - N Matrix (N):

         N = (X^T X + Λ)⁻¹ X^T

       - N Matrix Norms Squared (||Nᵍ||^2):

         For each group g:

         ||Nᵍ||^2 = Σ_{j ∈ g} |Nⱼ|^2

    3. M Squared Matrix (M^2):

       M^2 = (pₛ pₛ^T) / n²

       Where pₛ is the number of features in each group.
    """

    def __init__(self, rdg: BasicGroupRidgeWorkspace):
        self.groups = rdg.groups
        self.ps = np.array(rdg.groups.ps)
        self.n = rdg.n
        self.beta_norms_squared = np.array(
            rdg.groups.group_summary(rdg.beta_current, lambda x: np.sum(np.abs(x) ** 2))
        )
        N_matrix = rdg.XtXp_lambda_inv  # Use the (p, p) inverse matrix
        if N_matrix.shape[1] != self.ps.sum():
            raise ValueError(
                f"Length of N_matrix ({N_matrix.shape[1]}) does not match number of"
                f" features ({self.ps.sum()})"
            )
        self.N_norms_squared = np.array(
            rdg.groups.group_summary(N_matrix, lambda x: np.sum(np.abs(x) ** 2))
        )
        self.M_squared = np.outer(self.ps, self.ps) / self.n**2


def sigma_squared_path(
    rdg: BasicGroupRidgeWorkspace, mom: MomentTunerSetup, sigma_s_squared: np.ndarray
):
    """
    Compute the regularization path for different values of σ².

    This function evaluates how the Ridge regression coefficients and leave-one-out (LOO)
    errors change as σ² varies. By analyzing the regularization path, one
    can understand the impact of different levels of regularization on the model's performance.

    For each σ² in the provided array:

    1. Compute λ:

       λ = get_lambdas(mom, σ^2)

    2. Fit Ridge Model:

       Fit the Ridge regression model using the computed λ and evaluate:

       LOO Error = rdg.fit(λ)

    3. Store Coefficients:

       β = rdg.beta_current

    4. Aggregate Results:

       Collect the λ values, LOO errors, and β coefficients for each σ^2.

    Args:
        rdg (BasicGroupRidgeWorkspace): The Ridge regression workspace.
        mom (MomentTunerSetup): The moment tuner setup containing necessary statistics.
        sigma_s_squared (np.ndarray): An array of σ^2 values to evaluate.

    Returns:
        dict: A dictionary containing the regularization path information:
            - 'lambdas': Array of λ values for each σ^2.
            - 'loos': Array of LOO errors corresponding to each σ^2.
            - 'betas': Array of coefficient vectors corresponding to each σ^2.

    Raises:
        ValueError: If sigma_s_squared contains negative values.
    """
    if np.any(sigma_s_squared < 0):
        raise ValueError("sigma_s_squared values must be non-negative.")

    n_sigma = len(sigma_s_squared)
    n_groups = rdg.ngroups()
    loos_hat = np.zeros(n_sigma)
    lambdas = np.zeros((n_sigma, n_groups))
    betas = np.zeros((n_sigma, rdg.groups.p))

    for i, sigma_sq in enumerate(sigma_s_squared):
        try:
            lambdas_tmp = get_lambdas(mom, sigma_sq)
            lambdas[i, :] = lambdas_tmp
            loos_hat[i] = rdg.fit(lambdas_tmp)
            betas[i, :] = rdg.beta_current
        except RidgeRegressionError as e:
            print(f"Error at σ² = {sigma_sq}: {str(e)}")

    return {"lambdas": lambdas, "loos": loos_hat, "betas": betas}


def get_lambdas(mom: MomentTunerSetup, sigma_sq: float) -> np.ndarray:
    """
    Compute lambda values for a given sigma squared.

    This function calculates the regularization parameters (lambda) for each feature
    group based on moment-based statistics. The computed lambda balances the regularization
    strength across different groups, ensuring that groups with more features or higher variance
    receive appropriate penalization.

    1. Compute α²:

       α_g^2 = max(‖β_g‖^2 - σ² * ‖N_g‖^2, 0) / p_g

       for each group g, where:

       - ‖β_g‖^2 is the squared norm of the coefficients in group g.
       - ‖N_g‖^2 is the squared norm of the matrix N for group g.
       - p_g is the number of features in group g.

    2. Compute γ_s (gamma_s):

       γ_g = p_g / n

       where n is the number of samples.

    3. Compute λ (lambda):

       λ_g = (σ^2 * γ_g) / α_g^2

       for each group g.

    Args:
        mom (MomentTunerSetup): The moment tuner setup containing necessary statistics.
        sigma_sq (float): The σ^2 value for which to compute λ.

    Returns:
        np.ndarray: The computed λ values for each feature group.

    Raises:
        ValueError: If sigma_sq is negative.
        NumericalInstabilityError: If division by zero occurs during lambda computation.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")

    alpha_sq = get_alpha_s_squared(mom, sigma_sq)
    gamma_s = np.array(mom.ps) / mom.n

    LARGE_VALUE = 1e12

    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = sigma_sq * gamma_s / alpha_sq
        zero_alpha = alpha_sq == 0
        if np.any(zero_alpha):
            warnings.warn(
                f"alpha_sq has zero values for groups: {np.where(zero_alpha)[0]}. "
                "Assigning large lambda values to these groups."
            )
        lambdas = np.where(zero_alpha, LARGE_VALUE, lambdas)

    return lambdas


def get_alpha_s_squared(self, sigma_sq: float) -> np.ndarray:
    """
    Compute α_s² values for a given σ² using Non-Negative Least Squares (NNLS).

    This function calculates the α_s² values required for determining the
    regularization parameters (λ) in Ridge regression. The α_s² values
    encapsulate the balance between the coefficient norms and the influence of
    the design matrix, adjusted by σ².

    Steps:
    1. Compute Right-Hand Side (RHS):
       RHS_g = ‖β_g‖² - σ² * ‖N_g‖²
       for each group g.

    2. Solve the NNLS problem:
       min || M_squared * alpha_sq_by_p - RHS ||₂²
       subject to alpha_sq_by_p ≥ 0

    3. Compute α_g²:
       α_g² = alpha_sq_by_p * p_g

    Args:
        sigma_sq (float): The σ² value for which to compute α_s².

    Returns:
        np.ndarray: The computed α_s² values for each feature group.

    Raises:
        ValueError: If sigma_sq is negative or if any p_s is zero.
        NNLSError: If the NNLS algorithm fails to converge.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")
    if np.any(self.groups.ps == 0):
        raise ValueError("All p_s values must be non-zero.")

    # Compute the right-hand side
    rhs = self.beta_norms_squared - sigma_sq * self.N_norms_squared
    rhs = np.maximum(rhs, 0)

    # Solve the NNLS problem: M_squared * alpha_sq_by_p ≈ rhs
    try:
        alpha_sq_by_p = nonneg_lsq(self.M_squared, rhs, alg="fnnls")
    except NNLSError as e:
        raise NNLSError(f"Failed to compute alpha_s_squared: {str(e)}")

    alpha_s_squared = alpha_sq_by_p * self.groups.ps

    return alpha_s_squared
