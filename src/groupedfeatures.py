from typing import List, Callable, Union, TypeVar
import numpy as np

T = TypeVar('T')

class GroupedFeatures:
    """
    A class representing groups of features, wherein the first `ps[0]` features are one group,
    the next `ps[1]` features are the second group and so forth.
    """

    def __init__(self, ps: List[int]):
        self.ps = ps
        self.p = sum(ps)
        self.num_groups = len(ps)

    @classmethod
    def from_group_size(cls, group_size: int, num_groups: int):
        return cls([group_size] * num_groups)

    def ngroups(self) -> int:
        return self.num_groups

    def nfeatures(self) -> int:
        return self.p

    def group_idx(self, i: int) -> range:
        starts = np.cumsum([0] + self.ps[:-1])
        ends = np.cumsum(self.ps)
        return range(starts[i], ends[i])

    def group_summary(self, vec: List[T], f: Callable[[List[T]], T]) -> List[T]:
        starts = np.cumsum([0] + self.ps[:-1])
        ends = np.cumsum(self.ps)
        return [f(vec[start:end]) for start, end in zip(starts, ends)]

    def group_expand(self, vec_or_num: Union[List[T], T]) -> List[T]:
        if isinstance(vec_or_num, (int, float)):
            return [vec_or_num] * self.p
        
        arr = [0.0] * self.p
        for i in range(self.num_groups):
            idx = self.group_idx(i)
            arr[idx.start:idx.stop] = [vec_or_num[i]] * len(idx)
        return arr

# Helper function to mimic Julia's `fill` function
def fill(value: T, length: int) -> List[T]:
    return [value] * length