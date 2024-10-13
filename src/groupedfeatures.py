"""Create and manage grouped feature structures for statistical modeling."""

from typing import Callable, Union, TypeVar
import numpy as np

T = TypeVar("T")


class GroupedFeatures:
    r"""A class representing groups of features.

    The first :math:`ps[0]` features are one group, the next :math:`ps[1]` features are the
    second group, and so forth.

    Parameters
    ----------
    ps : list of int
        List of positive integers representing the size of each group.

    Attributes
    ----------
    ps : list of int
        List of group sizes.
    p : int
        Total number of features, calculated as :math:`p = \sum ps`.
    num_groups : int
        Number of groups, denoted as :math:`G`.

    Raises
    ------
    TypeError
        If `ps` is not a list or contains non-integer elements.
    ValueError
        If any group size is not positive.
    """

    def __init__(self, ps: list[int]):
        if not isinstance(ps, list):
            raise TypeError(
                f"ps must be a list of positive integers, got {type(ps).__name__}"
            )
        if not all(isinstance(p, int) for p in ps):
            raise TypeError("All group sizes in ps must be integers")
        if not all(p > 0 for p in ps):
            raise ValueError("All group sizes in ps must be positive integers")

        self.ps = ps
        self.p = sum(ps)
        self.num_groups = len(ps)

    @classmethod
    def from_group_size(cls, group_size: int, num_groups: int):
        r"""Create a `GroupedFeatures` instance with uniform group sizes.

        The group sizes are all set to :math:`group\_size`.

        Parameters
        ----------
        group_size : int
            Size of each group.
        num_groups : int
            Number of groups.

        Returns
        -------
        GroupedFeatures
            Instance with uniform group sizes.

        Raises
        ------
        TypeError
            If `group_size` or `num_groups` is not an integer.
        ValueError
            If `group_size` or `num_groups` is not positive.
        """
        if not isinstance(group_size, int):
            raise TypeError(
                f"group_size must be an integer, got {type(group_size).__name__}"
            )
        if not isinstance(num_groups, int):
            raise TypeError(
                f"num_groups must be an integer, got {type(num_groups).__name__}"
            )
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if num_groups <= 0:
            raise ValueError("num_groups must be a positive integer")

        return cls([group_size] * num_groups)

    def ngroups(self) -> int:
        r"""Get the number of groups.

        Returns
        -------
        int
            Number of groups, :math:`G`.
        """
        return self.num_groups

    def nfeatures(self) -> int:
        r"""
        Get the total number of features.

        Returns
        -------
        int
            Total number of features, :math:`p`.
        """
        return self.p

    def group_idx(self, i: int) -> range:
        r"""Get the range of feature indices for the :math:`i`-th group.

        Parameters
        ----------
        i : int
            Index of the group (0-based).

        Returns
        -------
        range
            Range object representing the indices of the group.

        Raises
        ------
        TypeError
            If `i` is not an integer.
        IndexError
            If `i` is out of range [0, :math:`num\_groups - 1`].
        """
        if not isinstance(i, int):
            raise TypeError(f"Group index i must be an integer, got {type(i).__name__}")
        if not (0 <= i < self.num_groups):
            raise IndexError(
                f"Group index i={i} is out of range [0, {self.num_groups -1}]"
            )

        starts = np.cumsum([0] + self.ps[:-1]).astype(int)
        ends = np.cumsum(self.ps).astype(int)
        return range(starts[i], ends[i])

    def group_summary(
        self,
        vec: Union[list[T], np.ndarray],
        f: Callable[[Union[list[T], np.ndarray]], T],
    ) -> list[T]:
        r"""Apply a summary function to each group of features.

        For each group :math:`g`, the function :math:`f` is applied to the subset of
        features in that group.

        Parameters
        ----------
        vec : array-like of shape (n_features,) or (n_samples, n_features)
            List or ndarray of features.
        f : callable
            Function that takes a list or ndarray of features and returns a
            summary value.

        Returns
        -------
        list
            List of summary values, one per group.

        Raises
        ------
        TypeError
            If `vec` is neither a list nor an ndarray, or if `f` is not callable.
        ValueError
            If length of `vec` does not match total number of features.
        """
        if not callable(f):
            raise TypeError("f must be a callable function")

        if isinstance(vec, np.ndarray):
            if vec.ndim == 2:
                if vec.shape[1] != self.p:
                    raise ValueError(
                        f"Length of vec ({vec.shape[1]}) does not match number of"
                        f" features ({self.p})"
                    )
            elif vec.ndim == 1:
                if vec.shape[0] != self.p:
                    raise ValueError(
                        f"Length of vec ({vec.shape[0]}) does not match number of"
                        f" features ({self.p})"
                    )
                vec = vec.reshape(1, -1)
            else:
                raise ValueError(f"vec must be 1D or 2D, got {vec.ndim}D")
        elif not isinstance(vec, list):
            raise TypeError(
                f"vec must be either a list or ndarray, got {type(vec).__name__}"
            )
        else:
            if len(vec) != self.p:
                raise ValueError(
                    f"Length of vec ({len(vec)}) does not match number of features"
                    f" ({self.p})"
                )

        summaries = []
        for i in range(self.num_groups):
            idx = self.group_idx(i)
            try:
                group_features = (
                    [vec[j] for j in idx]
                    if isinstance(vec, list)
                    else vec[:, idx.start : idx.stop]
                )
                summaries.append(f(group_features))
            except Exception as e:
                raise RuntimeError(
                    f"Error applying function to group {idx.start}-{idx.stop}: {e}"
                )

        return summaries

    def group_expand(self, vec_or_num: Union[list[T], T, np.ndarray]) -> list[T]:
        r"""
        Expand a vector or number to a list of features.

        If `vec_or_num` is a number, replicate it across all features.
        If it is a list or ndarray, each element corresponds to a group and is
        replicated within the group.

        Parameters
        ----------
        vec_or_num : int, float, list or ndarray
            Either a single number or a list/ndarray with length equal to number
            of groups.

        Returns
        -------
        list
            Expanded list of features.

        Raises
        ------
        TypeError
            If `vec_or_num` is neither a number nor a list/ndarray.
        ValueError
            If `vec_or_num` is a list/ndarray but its length does not match number
            of groups.
        """
        if isinstance(vec_or_num, (int, float)):
            return [vec_or_num] * self.p

        if isinstance(vec_or_num, (list, np.ndarray)):
            if len(vec_or_num) != self.num_groups:
                raise ValueError(
                    f"Length of vec_or_num ({len(vec_or_num)}) does not match number of"
                    f" groups ({self.num_groups})"
                )
            # If it's an ndarray, convert to list
            if isinstance(vec_or_num, np.ndarray):
                vec_or_num = vec_or_num.tolist()
            arr = [0.0] * self.p
            for i in range(self.num_groups):
                idx = self.group_idx(i)
                arr[idx.start : idx.stop] = [vec_or_num[i]] * len(idx)
            return arr

        raise TypeError(
            "vec_or_num must be either a number (int or float) or a list/ndarray, got"
            f" {type(vec_or_num).__name__}"
        )


def fill(value: T, length: int) -> list[T]:
    """Fill a list with a given value repeated :math:`length` times.

    Parameters
    ----------
    value : T
        Value to fill the list with.
    length : int
        Number of times to repeat the value.

    Returns
    -------
    list
        List containing the value repeated :math:`length` times.

    Raises
    ------
    TypeError
        If `length` is not an integer.
    ValueError
        If `length` is negative.
    """
    if not isinstance(length, int):
        raise TypeError(f"length must be an integer, got {type(length).__name__}")
    if length < 0:
        raise ValueError("length must be a non-negative integer")
    return [value] * length
