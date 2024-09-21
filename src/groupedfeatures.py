"""
This module provides a flexible and efficient way to handle grouped features. It introduces
the GroupedFeatures class, which represents a collection of features organized into groups
of varying sizes. This module is particularly useful in scenarios where features naturally
form groups, such as:
- Multi-modal data analysis
- Feature engineering with categorical variables
- Dealing with repeated measurements in time series data

Key components:

1. GroupedFeatures class:
   - Represents groups of features with customizable group sizes.
   - Supports initialization with a list of group sizes or a uniform group size.
   - Provides methods for accessing group information, summarizing groups, and expanding group-level data.

2. Utility functions:
   - fill: A helper function to create lists filled with a specific value.

Main features:

- Flexible group creation: Create feature groups with different sizes or uniform sizes.
- Group indexing: Easily access the index range for any group.
- Group summaries: Apply custom functions to summarize data within each group.
- Group expansion: Expand group-level data to feature-level data.
- Support for both list and numpy.ndarray inputs.
- Comprehensive error checking and informative error messages.

Usage example:
    gf = GroupedFeatures([2, 3, 4])  # Create groups of sizes 2, 3, and 4
    gf.ngroups()  # Returns 3
    gf.nfeatures()  # Returns 9
    gf.group_idx(1)  # Returns range(2, 5) for the second group
    gf.group_summary([1, 2, 3, 4, 5, 6, 7, 8, 9], sum)  # Returns [3, 12, 30]
    gf.group_expand([10, 20, 30])  # Returns [10, 10, 20, 20, 20, 30, 30, 30, 30]
"""

from typing import List, Callable, Union, TypeVar
import numpy as np

T = TypeVar('T')

class GroupedFeatures:
    """
    A class representing groups of features, wherein the first `ps[0]` features are one group,
    the next `ps[1]` features are the second group and so forth.
    """

    def __init__(self, ps: List[int]):
        """
        Initialize GroupedFeatures with a list of group sizes.

        :param ps: List of positive integers representing the size of each group.
        :raises TypeError: If ps is not a list or contains non-integer elements.
        :raises ValueError: If any group size is not positive.
        """
        if not isinstance(ps, list):
            raise TypeError(f"ps must be a list of positive integers, got {type(ps).__name__}")
        if not all(isinstance(p, int) for p in ps):
            raise TypeError("All group sizes in ps must be integers")
        if not all(p > 0 for p in ps):
            raise ValueError("All group sizes in ps must be positive integers")

        self.ps = ps
        self.p = sum(ps)
        self.num_groups = len(ps)

    @classmethod
    def from_group_size(cls, group_size: int, num_groups: int):
        """
        Create a GroupedFeatures instance with num_groups each of size group_size.

        :param group_size: Positive integer indicating the size of each group.
        :param num_groups: Positive integer indicating the number of groups.
        :return: GroupedFeatures instance.
        :raises TypeError: If group_size or num_groups is not an integer.
        :raises ValueError: If group_size or num_groups is not positive.
        """
        if not isinstance(group_size, int):
            raise TypeError(f"group_size must be an integer, got {type(group_size).__name__}")
        if not isinstance(num_groups, int):
            raise TypeError(f"num_groups must be an integer, got {type(num_groups).__name__}")
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if num_groups <= 0:
            raise ValueError("num_groups must be a positive integer")

        return cls([group_size] * num_groups)

    def ngroups(self) -> int:
        """
        Get the number of groups.

        :return: Number of groups.
        """
        return self.num_groups

    def nfeatures(self) -> int:
        """
        Get the total number of features.

        :return: Total number of features.
        """
        return self.p

    def group_idx(self, i: int) -> range:
        """
        Get the range of feature indices for the i-th group.

        :param i: Index of the group (0-based).
        :return: range object representing the indices of the group.
        :raises TypeError: If i is not an integer.
        :raises IndexError: If i is out of range [0, num_groups -1].
        """
        if not isinstance(i, int):
            raise TypeError(f"Group index i must be an integer, got {type(i).__name__}")
        if not (0 <= i < self.num_groups):
            raise IndexError(f"Group index i={i} is out of range [0, {self.num_groups -1}]")

        starts = np.cumsum([0] + self.ps[:-1]).astype(int)
        ends = np.cumsum(self.ps).astype(int)
        return range(starts[i], ends[i])

    def group_summary(
        self,
        vec: Union[List[T], np.ndarray],
        f: Callable[[Union[List[T], np.ndarray]], T]
    ) -> List[T]:
        """
        Apply a summary function to each group of features.

        :param vec: List or ndarray of features, length must be equal to total number of features.
        :param f: Callable that takes a list or ndarray of features and returns a summary value.
        :return: List of summary values, one per group.
        :raises TypeError: If vec is neither a list nor an ndarray, or if f is not callable.
        :raises ValueError: If length of vec does not match total number of features.
        """
        if not callable(f):
            raise TypeError("f must be a callable function")

        if isinstance(vec, np.ndarray):
            if vec.ndim == 2:
                if vec.shape[1] != self.p:
                    raise ValueError(f"Length of vec ({vec.shape[1]}) does not match number of features ({self.p})")
            elif vec.ndim == 1:
                if vec.shape[0] != self.p:
                    raise ValueError(f"Length of vec ({vec.shape[0]}) does not match number of features ({self.p})")
                vec = vec.reshape(1, -1)
            else:
                raise ValueError(f"vec must be 1D or 2D, got {vec.ndim}D")
        elif not isinstance(vec, list):
            raise TypeError(f"vec must be either a list or ndarray, got {type(vec).__name__}")
        else:
            if len(vec) != self.p:
                raise ValueError(f"Length of vec ({len(vec)}) does not match number of features ({self.p})")

        summaries = []
        for i in range(self.num_groups):
            idx = self.group_idx(i)
            try:
                group_features = [vec[j] for j in idx] if isinstance(vec, list) else vec[:, idx.start:idx.stop]
                summaries.append(f(group_features))
            except Exception as e:
                raise RuntimeError(f"Error applying function to group {idx.start}-{idx.stop}: {e}")

        return summaries

    def group_expand(self, vec_or_num: Union[List[T], T, np.ndarray]) -> List[T]:
        """
        Expand a vector or number to a list of features.

        If vec_or_num is a number (int or float), replicate it across all features.
        If it is a list or ndarray, each element corresponds to a group and is replicated within the group.

        :param vec_or_num: Either a single number or a list/ndarray with length equal to number of groups.
        :return: Expanded list of features.
        :raises TypeError: If vec_or_num is neither a number nor a list/ndarray.
        :raises ValueError: If vec_or_num is a list/ndarray but its length does not match number of groups.
        """
        if isinstance(vec_or_num, (int, float)):
            return [vec_or_num] * self.p

        if isinstance(vec_or_num, (list, np.ndarray)):
            if len(vec_or_num) != self.num_groups:
                raise ValueError(
                    f"Length of vec_or_num ({len(vec_or_num)}) does not match number of groups ({self.num_groups})"
                )
            # If it's an ndarray, convert to list
            if isinstance(vec_or_num, np.ndarray):
                vec_or_num = vec_or_num.tolist()
            arr = [0.0] * self.p
            for i in range(self.num_groups):
                idx = self.group_idx(i)
                arr[idx.start:idx.stop] = [vec_or_num[i]] * len(idx)
            return arr

        raise TypeError(
            f"vec_or_num must be either a number (int or float) or a list/ndarray, got {type(vec_or_num).__name__}"
        )

def fill(value: T, length: int) -> List[T]:
    """
    Fill a list with a given value repeated length times.

    :param value: Value to fill the list with.
    :param length: Number of times to repeat the value.
    :return: List containing the value repeated length times.
    :raises TypeError: If length is not an integer.
    :raises ValueError: If length is negative.
    """
    if not isinstance(length, int):
        raise TypeError(f"length must be an integer, got {type(length).__name__}")
    if length < 0:
        raise ValueError("length must be a non-negative integer")
    return [value] * length
