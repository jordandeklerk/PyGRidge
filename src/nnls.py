"""
NNLS Module
============

The `nnls.py` module within the PyGRidge package provides robust implementations for solving
Non-Negative Least Squares (NNLS) problems. NNLS is a fundamental optimization technique
widely used in various scientific and engineering disciplines, including data fitting, machine
learning, signal processing, and quantitative finance. This module offers flexible and efficient
algorithms to find solutions that adhere to non-negativity constraints, ensuring that the resulting
coefficients are physically interpretable and meaningful in practical applications.

Overview

Given a matrix A ∈ ℝ^(m × n) and a target matrix B ∈ ℝ^(m × k),
the goal is to find a solution matrix X ∈ ℝ^(n × k) that minimizes the Frobenius norm
of the residuals while enforcing non-negativity on X:

min_{X ≥ 0} || AX - B ||_F^2

Here, ||·||_F denotes the Frobenius norm, and the constraint X ≥ 0 ensures that
all elements of X are non-negative. This constraint is crucial in scenarios where negative
coefficients lack physical or practical significance, such as in concentration measurements,
image reconstruction, or portfolio optimization.

Key Components

1. Custom Exceptions

- NNLSError: Base exception class for all NNLS-related errors.
- InvalidInputError: Raised when the input matrices A or B are invalid, incompatible,
  or do not meet the necessary conditions for the algorithms to operate correctly.
- ConvergenceError: Raised when the algorithm fails to converge to a solution within the
  specified number of iterations.

2. Core Functions

a. nonneg_lsq

This function serves as the main entry point for solving NNLS problems. It provides a unified
interface to various NNLS algorithms, currently supporting the Fast Non-Negative Least Squares
(FNNLS) method. The function handles input validation, algorithm selection, and optional
parallelization for improved performance on multi-core systems.

b. fnnls

This function implements the Fast Non-Negative Least Squares algorithm, which is an efficient
method for solving large-scale NNLS problems. It supports both standard and Gram matrix inputs,
allowing for flexibility in problem formulation and potential computational optimizations.

c. fnnls_core

This is the core implementation of the FNNLS algorithm, designed to solve a single NNLS problem.
It employs an iterative approach with careful handling of numerical stability and convergence
issues.

Features and Capabilities

- Flexible input handling: Supports both standard matrix inputs and pre-computed Gram matrices.
- Parallelization: Offers optional parallel processing for multi-column target matrices.
- Robust error handling: Provides informative error messages and custom exceptions for various
  failure modes.
- Configurable parameters: Allows fine-tuning of tolerance levels, iteration limits, and other
  algorithm-specific parameters.
- Logging: Incorporates a logging system for tracking algorithm progress and debugging.
"""

import numpy as np
from typing import Union, Optional
from functools import partial
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NNLSError(Exception):
    """Base exception class for NNLS-related errors."""
    pass

class InvalidInputError(NNLSError):
    """Exception raised for invalid input to NNLS functions."""
    pass

class ConvergenceError(NNLSError):
    """Exception raised when the algorithm fails to converge."""
    pass

def nonneg_lsq(
    A: np.ndarray,
    B: Union[np.ndarray, np.ndarray],
    alg: str = 'fnnls',
    gram: bool = False,
    use_parallel: bool = False,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
    **kwargs
) -> np.ndarray:    
    """
    Solves the non-negative least squares problem.

    Parameters:
    ----------
    A : np.ndarray
        The input matrix A.
    B : np.ndarray
        The target matrix B. If B is a vector, it will be converted to a column matrix.
    alg : str, optional
        The algorithm to use. Currently supports 'fnnls'. Default is 'fnnls'.
    gram : bool, optional
        If True, A and B are treated as Gram matrices (AtA and AtB). Default is False.
    use_parallel : bool, optional
        If True and multiple CPUs are available, computations for multiple columns of B are parallelized. Default is False.
    tol : float, optional
        Tolerance for non-negativity constraints. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. If None, set to 30 * number of columns in AtA.
    **kwargs
        Additional keyword arguments passed to the underlying algorithm.

    Returns:
    -------
    np.ndarray
        Solution matrix X that minimizes ||A*X - B||_2 subject to X >= 0.
    
    Raises:
    ------
    InvalidInputError
        If the input matrices are invalid or incompatible.
    ValueError
        If the specified algorithm is not recognized.
    """
    try:
        # Input validation
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            raise InvalidInputError("A and B must be numpy arrays.")
        
        if A.size == 0 or B.size == 0:
            raise InvalidInputError("Input matrices A and B must not be empty.")
        
        if B.ndim == 1:
            B = B[:, np.newaxis]
        
        if not gram and A.shape[0] != B.shape[0]:
            raise InvalidInputError(f"Incompatible shapes: A has {A.shape[0]} rows, B has {B.shape[0]} rows.")
        
        if gram and A.shape[0] != A.shape[1]:
            raise InvalidInputError("When gram=True, A must be a square matrix.")
        
        if gram and A.shape[0] != B.shape[0]:
            raise InvalidInputError(f"Incompatible shapes for gram matrices: A has {A.shape[0]} rows, B has {B.shape[0]} rows.")
        
        if alg == 'fnnls':
            return fnnls(A, B, gram=gram, use_parallel=use_parallel, tol=tol, max_iter=max_iter, **kwargs)
        else:
            raise ValueError(f"Specified algorithm '{alg}' not recognized.")
    
    except NNLSError as e:
        logger.error(f"NNLS Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in nonneg_lsq: {str(e)}")
        raise

def fnnls(
    A: np.ndarray,
    B: Union[np.ndarray, np.ndarray],
    gram: bool = False,
    use_parallel: bool = False,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Solves the non-negative least squares problem using the FNNLS algorithm.

    Parameters:
    ----------
    A : np.ndarray
        The input matrix A or Gram matrix AtA if gram=True.
    B : np.ndarray
        The target matrix B or AtB if gram=True. If B is a vector, it will be converted to a column matrix.
    gram : bool, optional
        If True, A and B are treated as Gram matrices (AtA and AtB). Default is False.
    use_parallel : bool, optional
        If True and multiple CPUs are available, computations for multiple columns of B are parallelized. Default is False.
    tol : float, optional
        Tolerance for non-negativity constraints. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. If None, set to 30 * number of variables.
    **kwargs
        Additional keyword arguments passed to the core FNNLS algorithm.

    Returns:
    -------
    np.ndarray
        Solution matrix X that minimizes ||A*X - B||_2 subject to X >= 0.
    
    Raises:
    ------
    InvalidInputError
        If the input matrices are invalid or incompatible.
    ConvergenceError
        If the algorithm fails to converge within the maximum number of iterations.
    """
    try:
        if B.ndim == 1:
            B = B[:, np.newaxis]

        n, k = B.shape

        if gram:
            AtA = A
            AtB = B
        else:
            AtA = A.T @ A
            AtB = A.T @ B

        if max_iter is None:
            max_iter = 30 * AtA.shape[0]

        if use_parallel and cpu_count() > 1 and k > 1:
            # Define a partial function with fixed AtA and kwargs
            solve_fn = partial(fnnls_core, AtA, tol=tol, max_iter=max_iter, **kwargs)
            X = np.column_stack(
                Parallel(n_jobs=-1)(
                    delayed(solve_fn)(AtB[:, i]) for i in range(k)
                )
            )
        else:
            X = np.zeros_like(AtB)
            for i in range(k):
                X[:, i] = fnnls_core(AtA, AtB[:, i], tol=tol, max_iter=max_iter, **kwargs)
        
        if B.shape[1] == 1:
            return X.ravel()
        return X
    
    except NNLSError as e:
        logger.error(f"FNNLS Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fnnls: {str(e)}")
        raise

def fnnls_core(
    AtA: np.ndarray,
    Atb: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 300,
    **kwargs
) -> np.ndarray:
    """
    Core FNNLS algorithm to solve a single non-negative least squares problem.

    Parameters:
    ----------
    AtA : np.ndarray
        The Gram matrix A.T @ A.
    Atb : np.ndarray
        The product A.T @ b.
    tol : float, optional
        Tolerance for non-negativity constraints. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default is 300.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns:
    -------
    np.ndarray
        Solution vector x that minimizes ||A*x - b||_2 subject to x >= 0.
    
    Raises:
    ------
    ConvergenceError
        If the algorithm fails to converge within the maximum number of iterations.
    """
    try:
        n = AtA.shape[0]
        x = np.zeros(n, dtype=AtA.dtype)
        s = np.zeros(n, dtype=AtA.dtype)

        P = x > tol
        w = Atb - AtA @ x

        iter_count = 0

        while np.sum(P) < n and np.any(w[~P] > tol):
            # Mask w where P is False
            w_masked = np.where(~P, w, -np.inf)
            i = np.argmax(w_masked)
            if w_masked[i] == -np.inf:
                break  # No eligible index found
            P[i] = True

            # Solve least squares for variables in P
            AtA_P = AtA[np.ix_(P, P)]
            Atb_P = Atb[P]
            try:
                s_P = np.linalg.solve(AtA_P, Atb_P)
            except np.linalg.LinAlgError:
                s_P = np.linalg.lstsq(AtA_P, Atb_P, rcond=None)[0]
            
            s[P] = s_P
            s[~P] = 0.0

            # Inner loop: enforce non-negativity
            while np.any(s[P] <= tol):
                iter_count += 1
                if iter_count >= max_iter:
                    raise ConvergenceError(f"FNNLS failed to converge after {max_iter} iterations.")

                # Indices where s <= tol and P is True
                mask = (s <= tol) & P
                if not np.any(mask):
                    break

                ind = np.where(mask)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    alpha = np.min(x[ind] / (x[ind] - s[ind]))
                    alpha = np.minimum(alpha, 1.0)

                # Update x
                x += alpha * (s - x)
                x = np.maximum(x, 0.0)  # Ensure numerical stability

                # Remove variables where x is approximately zero
                P = x > tol

                # Recompute s for the new P
                AtA_P = AtA[np.ix_(P, P)]
                Atb_P = Atb[P]
                try:
                    s_P = np.linalg.solve(AtA_P, Atb_P)
                except np.linalg.LinAlgError:
                    s_P = np.linalg.lstsq(AtA_P, Atb_P, rcond=None)[0]
                
                s = np.zeros_like(s)
                s[P] = s_P

            x = s.copy()
            w = Atb - AtA @ x

        if iter_count >= max_iter:
            raise ConvergenceError(f"FNNLS failed to converge after {max_iter} iterations.")

        return x
    
    except ConvergenceError as e:
        logger.warning(f"Convergence issue in fnnls_core: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fnnls_core: {str(e)}")
        raise
