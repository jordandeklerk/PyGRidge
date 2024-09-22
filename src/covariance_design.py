"""
Covariance Design Module
========================

This module implements various covariance matrix designs. It provides a flexible framework for creating
and manipulating various covariance structures. The module defines abstract classes for covariance designs
and specific implementations for different types of covariance matrices.
Key components:

1. DiscreteNonParametric: A class representing discrete non-parametric spectra with eigenvalues and associated probabilities.

2. CovarianceDesign: An abstract base class defining the interface for all covariance matrix designs.

3. Specific covariance designs:
   - AR1Design: Implements an AutoRegressive model of order 1.
   - DiagonalCovarianceDesign: An abstract base class for diagonal covariance matrices.
   - IdentityCovarianceDesign: Constructs an identity covariance matrix.
   - UniformScalingCovarianceDesign: Creates a diagonal matrix with uniform scaling.
   - ExponentialOrderStatsCovarianceDesign: Generates eigenvalues based on exponential order statistics.

4. BlockDiagonal: A class for representing block diagonal matrices.

5. MixtureModel: Represents a mixture of multiple spectra with associated mixing proportions.

6. BlockCovarianceDesign: Constructs complex covariance structures by composing multiple covariance designs.

7. Utility functions:
   - block_diag: Constructs a block diagonal matrix from input arrays.
   - simulate_rotated_design: Simulates a rotated design matrix based on a given covariance design.
   - set_groups: Configures feature groups for covariance designs.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Union
import numpy as np

from PyGRidge.src.groupedfeatures import GroupedFeatures, fill


class DiscreteNonParametric:
    """
    Represents a discrete non-parametric spectrum characterized by eigenvalues and their associated probabilities.

    Parameters:
        eigs (List[float]): A list of eigenvalues.
        probs (List[float]): A list of probabilities corresponding to each eigenvalue.

    Attributes:
        eigs (np.ndarray): Numpy array of eigenvalues.
        probs (np.ndarray): Numpy array of probabilities for each eigenvalue.

    Example:
        >>> spectrum = DiscreteNonParametric(eigs=[1.0, 2.0, 3.0], probs=[0.2, 0.3, 0.5])
    """
    def __init__(self, eigs: List[float], probs: List[float]):
        if not isinstance(eigs, list):
            raise TypeError(f"'eigs' must be a list of floats, got {type(eigs).__name__}")
        if not all(isinstance(e, (int, float)) for e in eigs):
            raise ValueError("All elements in 'eigs' must be integers or floats.")
        if not isinstance(probs, list):
            raise TypeError(f"'probs' must be a list of floats, got {type(probs).__name__}")
        if not all(isinstance(p, (int, float)) for p in probs):
            raise ValueError("All elements in 'probs' must be integers or floats.")
        if len(eigs) != len(probs):
            raise ValueError("'eigs' and 'probs' must be of the same length.")
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("The probabilities in 'probs' must sum to 1.")
        if any(p < 0 for p in probs):
            raise ValueError("Probabilities in 'probs' must be non-negative.")

        self.eigs = np.array(eigs)
        self.probs = np.array(probs)


class CovarianceDesign(ABC):
    """
    Abstract base class defining the interface for covariance matrix designs.

    This class outlines the essential methods that any covariance design must implement:
    - `get_Sigma`: Retrieve the covariance matrix.
    - `nfeatures`: Obtain the number of features (dimensions) in the covariance matrix.
    - `spectrum`: Get the spectral representation of the covariance matrix.

    Methods:
        get_Sigma() -> np.ndarray:
            Returns the covariance matrix Σ.

        nfeatures() -> int:
            Returns the number of features (size of Σ).

        spectrum() -> DiscreteNonParametric:
            Returns the spectrum of Σ as a DiscreteNonParametric object.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_Sigma(self) -> np.ndarray:
        """
        Constructs and returns the covariance matrix Σ.

        Returns:
            np.ndarray: The covariance matrix.
        """
        pass

    @abstractmethod
    def nfeatures(self) -> int:
        """
        Retrieves the number of features (dimensions) in the covariance matrix Σ.

        Returns:
            int: Number of features.
        """
        pass

    @abstractmethod
    def spectrum(self) -> DiscreteNonParametric:
        """
        Computes the spectral decomposition of the covariance matrix Σ.

        Returns:
            DiscreteNonParametric: Spectrum containing eigenvalues and their probabilities.
        """
        pass


class AR1Design(CovarianceDesign):
    """
    Implements an AutoRegressive model of order 1 (AR(1)) for covariance matrix design.

    In an AR(1) model, each element Σ_{i,j} of the covariance matrix Σ is defined as:

    Σ_{i,j} = ρ^{|i - j|}

    where ρ is the correlation coefficient between adjacent features.

    Parameters:
        p (int, optional): Number of features (dimensions). Must be set before generating Σ.
        rho (float, default=0.7): The AR(1) parameter representing the correlation coefficient.

    Attributes:
        p (int): Number of features.
        rho (float): AR(1) correlation coefficient.

    Methods:
        get_Sigma() -> np.ndarray:
            Constructs the AR(1) covariance matrix Σ.

        nfeatures() -> int:
            Returns the number of features p.

        spectrum() -> DiscreteNonParametric:
            Computes the eigenvalues of Σ and assigns equal probability to each.
    """
    def __init__(self, p: int = None, rho: float = 0.7):
        super().__init__()
        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"'p' must be an integer, got {type(p).__name__}")
            if p <= 0:
                raise ValueError("'p' must be a positive integer.")
        if not isinstance(rho, (int, float)):
            raise TypeError(f"'rho' must be a float, got {type(rho).__name__}")
        if not (0 <= rho < 1):
            raise ValueError("'rho' must be in the interval [0, 1).")
        self.p = p
        self.rho = rho

    def get_Sigma(self) -> np.ndarray:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        rho = self.rho
        indices = np.arange(self.p)

        # Compute |i - j| for all i,j
        try:
            Sigma = rho ** np.abs(np.subtract.outer(indices, indices))
        except Exception as e:
            raise RuntimeError(f"Failed to compute covariance matrix: {e}")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Covariance matrix Σ must be symmetric.")
        return Sigma

    def nfeatures(self) -> int:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if not isinstance(self.p, int) or self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        return self.p

    def spectrum(self) -> DiscreteNonParametric:
        Sigma = self.get_Sigma()
        try:
            eigs = np.linalg.eigvals(Sigma)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Eigenvalue computation failed: {e}")
        if eigs.size == 0:
            raise ValueError("Covariance matrix Σ has no eigenvalues.")
        probs = fill(1.0 / len(eigs), len(eigs))
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1.")
        return DiscreteNonParametric(eigs.tolist(), probs)


class DiagonalCovarianceDesign(CovarianceDesign):
    """
    Abstract base class for covariance designs that produce diagonal covariance matrices.

    Since diagonal covariance matrices have non-zero entries only on the diagonal, this class
    provides a common structure for such designs, managing the number of features.

    Parameters:
        p (int, optional): Number of features (dimensions). Must be set before generating Σ.

    Attributes:
        p (int): Number of features.

    Methods:
        nfeatures() -> int:
            Returns the number of features p.
    """
    def __init__(self, p: int = None):
        super().__init__()
        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"'p' must be an integer, got {type(p).__name__}")
            if p <= 0:
                raise ValueError("'p' must be a positive integer.")
        self.p = p

    def nfeatures(self) -> int:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if not isinstance(self.p, int) or self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        return self.p


class IdentityCovarianceDesign(DiagonalCovarianceDesign):
    """
    Constructs an identity covariance matrix Σ, where all diagonal entries are 1 and off-diagonal entries are 0.

    The identity matrix represents uncorrelated features with unit variance.

    Parameters:
        p (int, optional): Number of features (dimensions). Must be set before generating Σ.

    Attributes:
        p (int): Number of features.

    Methods:
        get_Sigma() -> np.ndarray:
            Returns the identity matrix of size p × p.

        spectrum() -> DiscreteNonParametric:
            Returns a spectrum where all eigenvalues are 1 with equal probabilities.
    """
    def __init__(self, p: int = None):
        super().__init__(p)

    def get_Sigma(self) -> np.ndarray:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        try:
            Sigma = np.identity(self.p)
        except Exception as e:
            raise RuntimeError(f"Failed to create identity matrix: {e}")
        return Sigma

    def spectrum(self) -> DiscreteNonParametric:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        eigs = [1.0] * self.p
        probs = [1.0 / self.p] * self.p
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1.")
        return DiscreteNonParametric(eigs, probs)


class UniformScalingCovarianceDesign(DiagonalCovarianceDesign):
    """
    Constructs a diagonal covariance matrix Σ with uniform scaling on the diagonal.

    Each diagonal entry Σ_{i,i} is set to a constant scaling factor.

    Parameters:
        scaling (float, default=1.0): The scaling factor applied uniformly to all diagonal entries.
        p (int, optional): Number of features (dimensions). Must be set before generating Σ.

    Attributes:
        scaling (float): Scaling factor for the diagonal entries.
        p (int): Number of features.

    Methods:
        get_Sigma() -> np.ndarray:
            Returns a diagonal matrix with each diagonal entry equal to `scaling`.

        spectrum() -> DiscreteNonParametric:
            Returns a spectrum where all eigenvalues are equal to `scaling` with equal probabilities.
    """
    def __init__(self, scaling: float = 1.0, p: int = None):
        super().__init__(p)
        if not isinstance(scaling, (int, float)):
            raise TypeError(f"'scaling' must be a float, got {type(scaling).__name__}")
        if scaling <= 0:
            raise ValueError("'scaling' must be a positive number.")
        self.scaling = scaling

    def get_Sigma(self) -> np.ndarray:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        try:
            Sigma = self.scaling * np.identity(self.p)
        except Exception as e:
            raise RuntimeError(f"Failed to create scaled identity matrix: {e}")
        return Sigma

    def spectrum(self) -> DiscreteNonParametric:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        eigs = [self.scaling] * self.p
        probs = [1.0 / self.p] * self.p
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1.")
        return DiscreteNonParametric(eigs, probs)


class ExponentialOrderStatsCovarianceDesign(DiagonalCovarianceDesign):
    """
    Constructs a diagonal covariance matrix Σ using exponential order statistics for the eigenvalues.
    The eigenvalues are generated based on the order statistics of exponential random variables with a specified rate.

    The i-th eigenvalue is computed as:

    λ_i = (1 / rate) * log(1 / t_i)

    where t_i are uniformly spaced points in the interval (1 / (2p), 1 - 1 / (2p)).

    Parameters:
        p (int, optional): Number of features (dimensions). Must be set before generating Σ.
        rate (float, default=1.0): Rate parameter for the exponential distribution.

    Attributes:
        rate (float): Rate parameter for generating eigenvalues.
        p (int): Number of features.

    Methods:
        spectrum() -> DiscreteNonParametric:
            Generates the eigenvalues based on exponential order statistics and assigns equal probabilities.

        get_Sigma() -> np.ndarray:
            Returns a diagonal matrix with the generated eigenvalues.
    """
    def __init__(self, p: int = None, rate: float = 1.0):
        super().__init__(p)
        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"'p' must be an integer, got {type(p).__name__}")
            if p <= 0:
                raise ValueError("'p' must be a positive integer.")
        if not isinstance(rate, (int, float)):
            raise TypeError(f"'rate' must be a float, got {type(rate).__name__}")
        if rate <= 0:
            raise ValueError("'rate' must be a positive number.")
        self.rate = rate

    def spectrum(self) -> DiscreteNonParametric:
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if self.p <= 0:
            raise ValueError("'p' must be a positive integer.")
        p = self.p
        rate = self.rate
        try:
            tmp = np.linspace(1 / (2 * p), 1 - 1 / (2 * p), p)
            eigs = (1 / rate) * np.log(1 / tmp)
        except Exception as e:
            raise RuntimeError(f"Failed to compute eigenvalues: {e}")
        probs = fill(1.0 / p, p)
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1.")
        if len(eigs) != p:
            raise ValueError("Number of eigenvalues must match the number of features.")
        return DiscreteNonParametric(eigs.tolist(), probs)

    def get_Sigma(self) -> np.ndarray:
        spectrum = self.spectrum()
        try:
            Sigma = np.diag(spectrum.eigs)
        except Exception as e:
            raise RuntimeError(f"Failed to create diagonal matrix from eigenvalues: {e}")
        if Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Covariance matrix Σ must be square.")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Covariance matrix Σ must be symmetric.")
        return Sigma


class BlockDiagonal:
    """
    Represents a block diagonal matrix composed of smaller square matrices (blocks).

    The overall covariance matrix Σ is constructed by placing each block along the diagonal,
    with all off-diagonal blocks being zero matrices.

    For example, given blocks B_1, B_2, ..., B_k, the block diagonal matrix Σ is:

    Σ = [B_1   0   ...   0  ]
        [0    B_2  ...   0  ]
        [.     .   .     .  ]
        [.     .    .    .  ]
        [.     .     .   .  ]
        [0     0   ...  B_k ]

    Parameters:
        blocks (List[np.ndarray]): A list of square numpy arrays to be placed on the diagonal.

    Attributes:
        blocks (List[np.ndarray]): List of block matrices.

    Methods:
        get_Sigma() -> np.ndarray:
            Constructs and returns the block diagonal matrix Σ.
    """
    def __init__(self, blocks: List[np.ndarray]):
        if not isinstance(blocks, list):
            raise TypeError(f"'blocks' must be a list of numpy arrays, got {type(blocks).__name__}")
        if not all(isinstance(block, np.ndarray) for block in blocks):
            raise TypeError("All elements in 'blocks' must be numpy.ndarray instances.")
        if not all(block.ndim == 2 for block in blocks):
            raise ValueError("All blocks must be 2-dimensional numpy arrays.")
        if not all(block.shape[0] == block.shape[1] for block in blocks):
            raise ValueError("All blocks must be square matrices.")
        self.blocks = blocks

    def get_Sigma(self) -> np.ndarray:
        try:
            Sigma = block_diag(*self.blocks)
        except Exception as e:
            raise RuntimeError(f"Failed to construct block diagonal matrix: {e}")
        if Sigma.size == 0:
            raise ValueError("Resulting covariance matrix Σ is empty.")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Covariance matrix Σ must be symmetric.")
        return Sigma


def block_diag(*arrs):
    """
    Constructs a block diagonal matrix from the provided input arrays.

    If no arrays are provided, returns an empty 2D array.

    Parameters:
        *arrs (np.ndarray): Variable number of square 2D numpy arrays to be placed on the diagonal.

    Returns:
        np.ndarray: The resulting block diagonal matrix.

    Example:
        >>> A = np.array([[1, 0], [0, 1]])
        >>> B = np.array([[2, 0], [0, 2]])
        >>> block_diag(A, B)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2]])
    """
    if not all(isinstance(a, np.ndarray) for a in arrs):
        raise TypeError("All arguments must be numpy.ndarray instances.")
    if not all(a.ndim == 2 for a in arrs):
        raise ValueError("All input arrays must be 2-dimensional.")
    if not all(a.shape[0] == a.shape[1] for a in arrs):
        raise ValueError("All input arrays must be square matrices.")
    if len(arrs) == 0:
        return np.array([[]])
    try:
        shapes = np.array([a.shape for a in arrs])
        out_shape = np.sum(shapes, axis=0)
        out = np.zeros(out_shape, dtype=arrs[0].dtype)
        r, c = 0, 0
        for a in arrs:
            out[r:r+a.shape[0], c:c+a.shape[1]] = a
            r += a.shape[0]
            c += a.shape[1]
    except Exception as e:
        raise RuntimeError(f"Failed to construct block diagonal matrix: {e}")
    return out


class MixtureModel:
    """
    Represents a mixture model consisting of multiple spectra, each weighted by a mixing proportion.

    The mixture model Σ is defined as:

    Σ = Σ_{i=1}^k π_i * Σ_i

    where Σ_i are individual covariance matrices (spectra) and π_i are their corresponding mixing proportions.

    Parameters:
        spectra (List[DiscreteNonParametric]): A list of spectra that comprise the mixture.
        mixing_prop (List[float]): A list of mixing proportions corresponding to each spectrum.

    Attributes:
        spectra (List[DiscreteNonParametric]): List of component spectra.
        mixing_prop (List[float]): Mixing proportions for each spectrum.

    Example:
        >>> spectrum1 = DiscreteNonParametric(eigs=[1, 2], probs=[0.5, 0.5])
        >>> spectrum2 = DiscreteNonParametric(eigs=[3, 4], probs=[0.3, 0.7])
        >>> mixture = MixtureModel(spectra=[spectrum1, spectrum2], mixing_prop=[0.6, 0.4])
    """
    def __init__(self, spectra: List[DiscreteNonParametric], mixing_prop: List[float]):
        if not isinstance(spectra, list):
            raise TypeError(f"'spectra' must be a list of DiscreteNonParametric instances, got {type(spectra).__name__}")
        if not all(isinstance(s, DiscreteNonParametric) for s in spectra):
            raise TypeError("All elements in 'spectra' must be instances of DiscreteNonParametric.")
        if not isinstance(mixing_prop, list):
            raise TypeError(f"'mixing_prop' must be a list of floats, got {type(mixing_prop).__name__}")
        if not all(isinstance(p, (int, float)) for p in mixing_prop):
            raise ValueError("All elements in 'mixing_prop' must be integers or floats.")
        if len(spectra) != len(mixing_prop):
            raise ValueError("'spectra' and 'mixing_prop' must be of the same length.")
        if not np.isclose(sum(mixing_prop), 1.0):
            raise ValueError("The mixing proportions in 'mixing_prop' must sum to 1.")
        if any(p < 0 for p in mixing_prop):
            raise ValueError("Mixing proportions in 'mixing_prop' must be non-negative.")

        self.spectra = spectra
        self.mixing_prop = mixing_prop


class BlockCovarianceDesign(CovarianceDesign):
    """
    Constructs a block covariance matrix by composing multiple covariance designs.

    Each block within the block diagonal structure can have its own covariance characteristics,
    allowing for complex covariance structures composed of simpler sub-components.

    Parameters:
        blocks (List[CovarianceDesign]): A list of covariance design instances, each representing a block.
        groups (GroupedFeatures, optional): An instance specifying feature groupings. If provided,
            it adjusts the mixing proportions based on group sizes.

    Attributes:
        blocks (List[CovarianceDesign]): List of covariance designs for each block.
        groups (GroupedFeatures, optional): Feature groupings affecting mixing proportions.

    Methods:
        get_Sigma() -> np.ndarray:
            Constructs and returns the block diagonal covariance matrix Σ.

        nfeatures() -> int:
            Returns the total number of features across all blocks.

        spectrum() -> MixtureModel:
            Combines the spectra of all blocks into a single mixture model, adjusting mixing proportions
            based on group sizes if `groups` is provided.
    """
    def __init__(self, blocks: List[CovarianceDesign], groups: GroupedFeatures = None):
        super().__init__()
        if not isinstance(blocks, list):
            raise TypeError(f"'blocks' must be a list of CovarianceDesign instances, got {type(blocks).__name__}")
        if not all(isinstance(block, CovarianceDesign) for block in blocks):
            raise TypeError("All elements in 'blocks' must be instances of CovarianceDesign.")
        if groups is not None and not isinstance(groups, GroupedFeatures):
            raise TypeError(f"'groups' must be an instance of GroupedFeatures or None, got {type(groups).__name__}")
        self.blocks = blocks
        self.groups = groups
        if self.groups is not None:
            if len(self.groups.ps) != len(self.blocks):
                raise ValueError("Number of groups must match number of blocks in BlockCovarianceDesign.")

    def get_Sigma(self) -> np.ndarray:
        try:
            block_matrices = [block.get_Sigma() for block in self.blocks]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve covariance matrices from blocks: {e}")
        try:
            Sigma = block_diag(*block_matrices)
        except Exception as e:
            raise RuntimeError(f"Failed to construct block diagonal covariance matrix: {e}")
        if Sigma.size == 0:
            raise ValueError("Resulting covariance matrix Σ is empty.")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Covariance matrix Σ must be symmetric.")
        return Sigma

    def nfeatures(self) -> int:
        try:
            total_features = sum(block.nfeatures() for block in self.blocks)
        except Exception as e:
            raise RuntimeError(f"Failed to compute total number of features: {e}")
        if total_features <= 0:
            raise ValueError("Total number of features must be positive.")
        return total_features

    def spectrum(self) -> MixtureModel:
        try:
            spectra = [block.spectrum() for block in self.blocks]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve spectra from blocks: {e}")
        if self.groups is not None:
            if len(self.groups.ps) != len(spectra):
                raise ValueError("Number of groups must match number of blocks.")
            mixing_prop = []
            total_p = self.groups.p
            if total_p == 0:
                raise ValueError("Total number of features across all groups must be positive.")
            for ps in self.groups.ps:
                if not isinstance(ps, int):
                    raise TypeError("Group sizes in 'groups.ps' must be integers.")
                if ps < 0:
                    raise ValueError("Group sizes in 'groups.ps' must be non-negative.")
                mixing_prop.append(ps / total_p)
        else:
            num_spectra = len(spectra)
            if num_spectra == 0:
                raise ValueError("There must be at least one spectrum to form a mixture model.")
            mixing_prop = [1.0 / num_spectra] * num_spectra
        if not np.isclose(sum(mixing_prop), 1.0):
            raise ValueError("Mixing proportions must sum to 1.")
        return MixtureModel(spectra, mixing_prop)


def simulate_rotated_design(cov: CovarianceDesign, n: int, rotated_measure: Callable = None) -> np.ndarray:
    """
    Simulates a rotated design matrix based on the provided covariance design.

    The simulation generates `n` samples from a multivariate distribution with covariance matrix Σ
    specified by the `CovarianceDesign` instance. The rotation is achieved using the Cholesky decomposition
    of Σ, i.e., Σ = LL^T, where L is a lower triangular matrix.

    The simulated design matrix X is computed as:

    X = ZL

    where Z is an n × p matrix with independent entries generated by `rotated_measure`.

    Parameters:
        cov (CovarianceDesign): An instance of CovarianceDesign defining Σ.
        n (int): Number of samples to generate.
        rotated_measure (Callable, optional): A callable that generates random samples. Defaults to `np.random.normal`.

    Returns:
        np.ndarray: An n × p simulated design matrix adhering to the covariance structure Σ.

    Raises:
        ValueError: If Σ is not positive definite, making the Cholesky decomposition impossible.
        TypeError: If input types are incorrect.
    Example:
        >>> ar1 = AR1Design(p=5, rho=0.7)
        >>> X = simulate_rotated_design(ar1, n=100)
    """
    if not isinstance(cov, CovarianceDesign):
        raise TypeError(f"'cov' must be an instance of CovarianceDesign, got {type(cov).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"'n' must be an integer, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("'n' must be a positive integer.")
    if rotated_measure is not None and not callable(rotated_measure):
        raise TypeError("'rotated_measure' must be a callable or None.")

    if rotated_measure is None:
        rotated_measure = np.random.normal

    Sigma = cov.get_Sigma()
    if not isinstance(Sigma, np.ndarray):
        raise TypeError("Covariance matrix Σ must be a numpy.ndarray.")
    if Sigma.ndim != 2:
        raise ValueError("Covariance matrix Σ must be 2-dimensional.")
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Covariance matrix Σ must be square.")
    if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
        raise ValueError("Covariance matrix Σ contains NaN or infinite values.")

    try:
        Sigma_chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma is not positive definite; Cholesky decomposition failed.")

    p = cov.nfeatures()
    if not isinstance(p, int):
        raise TypeError("'nfeatures()' must return an integer.")
    if p <= 0:
        raise ValueError("Number of features 'p' must be positive.")

    try:
        Z = rotated_measure(size=(n, p))
    except Exception as e:
        raise RuntimeError(f"Failed to generate random samples using 'rotated_measure': {e}")

    if not isinstance(Z, np.ndarray):
        raise TypeError("Output from 'rotated_measure' must be a numpy.ndarray.")
    if Z.shape != (n, p):
        raise ValueError(f"Shape of Z must be ({n}, {p}), got {Z.shape}.")

    try:
        X = Z @ Sigma_chol
    except Exception as e:
        raise RuntimeError(f"Failed to compute the simulated design matrix X: {e}")

    if not isinstance(X, np.ndarray):
        raise TypeError("Simulated design matrix X must be a numpy.ndarray.")
    if X.shape != (n, p):
        raise ValueError(f"Shape of simulated design matrix X must be ({n}, {p}), got {X.shape}.")

    return X


def set_groups(design: CovarianceDesign, groups_or_p: Union[GroupedFeatures, int]):
    """
    Configures the number of features or feature groups for a given `CovarianceDesign` instance.

    This function allows setting the dimensionality either as a single integer (total number of features)
    or as a `GroupedFeatures` instance that specifies groupings within the features. When provided
    with group information, the covariance design adjusts its internal structure to accommodate the groups.

    Parameters:
        design (CovarianceDesign): An instance of CovarianceDesign to configure.
        groups_or_p (GroupedFeatures or int): Either an integer specifying the total number of features
            or a `GroupedFeatures` instance defining feature groupings.

    Raises:
        TypeError: If `groups_or_p` is neither an instance of `GroupedFeatures` nor an integer.
        ValueError: If group sizes are inconsistent with the number of blocks in `BlockCovarianceDesign`.
    Example:
        >>> groups = GroupedFeatures(ps=[2, 3])
        >>> set_groups(design, groups)
    """
    if not isinstance(design, CovarianceDesign):
        raise TypeError(f"'design' must be an instance of CovarianceDesign, got {type(design).__name__}")

    if isinstance(groups_or_p, GroupedFeatures):
        groups = groups_or_p
        p = groups.nfeatures()
        if isinstance(design, BlockCovarianceDesign):
            if len(groups.ps) != len(design.blocks):
                raise ValueError("Number of groups must match number of blocks in BlockCovarianceDesign.")
            for block, ps in zip(design.blocks, groups.ps):
                set_groups(block, ps)
            design.groups = groups
        else:
            design.p = p
    elif isinstance(groups_or_p, int):
        if groups_or_p <= 0:
            raise ValueError("'groups_or_p' as an integer must be a positive value.")
        if isinstance(design, BlockCovarianceDesign):
            for block in design.blocks:
                set_groups(block, groups_or_p)
        else:
            design.p = groups_or_p
    else:
        raise TypeError("groups_or_p must be an instance of GroupedFeatures or int.")