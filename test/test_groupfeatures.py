import pytest
from typing import List, Callable

from ..src.groupedfeatures import GroupedFeatures, fill

# Define a simple summary function for testing
def sum_func(group: List[int]) -> int:
    return sum(group)

def invalid_func(group: List[int]) -> int:
    raise ValueError("Invalid function for testing")

class TestGroupedFeatures:
    # Tests for __init__

    def test_init_valid(self):
        ps = [2, 3, 5]
        gf = GroupedFeatures(ps)
        assert gf.ps == ps
        assert gf.p == 10
        assert gf.num_groups == 3

    @pytest.mark.parametrize("ps, expected_exception, message", [
        (None, TypeError, "ps must be a list"),
        ("not a list", TypeError, "ps must be a list"),
        ([1.5, 2, 3], TypeError, "All group sizes in ps must be integers"),
        ([1, -2, 3], ValueError, "All group sizes in ps must be positive integers"),
        ([0, 1, 2], ValueError, "All group sizes in ps must be positive integers"),
    ])
    def test_init_invalid(self, ps, expected_exception, message):
        with pytest.raises(expected_exception, match=message):
            GroupedFeatures(ps)

    # Tests for from_group_size

    def test_from_group_size_valid(self):
        group_size = 4
        num_groups = 5
        gf = GroupedFeatures.from_group_size(group_size, num_groups)
        assert gf.ps == [4] * 5
        assert gf.p == 20
        assert gf.num_groups == 5

    @pytest.mark.parametrize("group_size, num_groups, expected_exception, message", [
        ("4", 5, TypeError, "group_size must be an integer"),
        (4, "5", TypeError, "num_groups must be an integer"),
        (0, 5, ValueError, "group_size must be a positive integer"),
        (-1, 5, ValueError, "group_size must be a positive integer"),
        (4, 0, ValueError, "num_groups must be a positive integer"),
        (4, -2, ValueError, "num_groups must be a positive integer"),
    ])
    def test_from_group_size_invalid(self, group_size, num_groups, expected_exception, message):
        with pytest.raises(expected_exception, match=message):
            GroupedFeatures.from_group_size(group_size, num_groups)

    # Tests for ngroups and nfeatures

    def test_ngroups_nfeatures(self):
        ps = [1, 2, 3]
        gf = GroupedFeatures(ps)
        assert gf.ngroups() == 3
        assert gf.nfeatures() == 6

    # Tests for group_idx

    @pytest.mark.parametrize("ps, i, expected_range", [
        ([2, 3, 5], 0, range(0,2)),
        ([2, 3, 5], 1, range(2,5)),
        ([2, 3, 5], 2, range(5,10)),
    ])
    def test_group_idx_valid(self, ps, i, expected_range):
        gf = GroupedFeatures(ps)
        idx = gf.group_idx(i)
        assert list(idx) == list(expected_range)

    @pytest.mark.parametrize("ps, i, expected_exception, message", [
        ([2, 3, 5], "1", TypeError, "Group index i must be an integer"),
        ([2, 3, 5], 3, IndexError, "Group index i=3 is out of range"),
        ([2, 3, 5], -1, IndexError, "Group index i=-1 is out of range"),
        ([2, 3, 5], 100, IndexError, "Group index i=100 is out of range"),
    ])
    def test_group_idx_invalid(self, ps, i, expected_exception, message):
        gf = GroupedFeatures(ps)
        with pytest.raises(expected_exception, match=message):
            gf.group_idx(i)

    # Tests for group_summary

    def test_group_summary_valid(self):
        ps = [2, 3]
        gf = GroupedFeatures(ps)
        vec = [1, 2, 3, 4, 5]
        summary = gf.group_summary(vec, sum_func)
        assert summary == [3, 12]  # sum([1,2])=3 and sum([3,4,5])=12

    @pytest.mark.parametrize("ps, vec, f, expected_exception, message", [
        ([2, 3], "not a list", sum_func, TypeError, "vec must be either a list or ndarray, got str"),
        ([2, 3], [1, 2], sum_func, ValueError, "Length of vec \\(2\\) does not match number of features \\(5\\)"),
        ([2, 3], [1, 2, 3, 4, 5], "not callable", TypeError, "f must be a callable function"),
    ])
    def test_group_summary_invalid(self, ps, vec, f, expected_exception, message):
        gf = GroupedFeatures(ps)
        with pytest.raises(expected_exception, match=message):
            gf.group_summary(vec, f)

    def test_group_summary_function_exception(self):
        ps = [2, 3]
        gf = GroupedFeatures(ps)
        vec = [1, 2, 3, 4, 5]
        with pytest.raises(RuntimeError, match="Error applying function to group 0-2: Invalid function for testing"):
            gf.group_summary(vec, invalid_func)

    # Tests for group_expand

    def test_group_expand_with_number(self):
        ps = [2, 3]
        gf = GroupedFeatures(ps)
        num = 7
        expanded = gf.group_expand(num)
        assert expanded == [7, 7, 7, 7, 7]

    def test_group_expand_with_list(self):
        ps = [2, 3]
        gf = GroupedFeatures(ps)
        vec_or_num = [10, 20]
        expanded = gf.group_expand(vec_or_num)
        assert expanded == [10, 10, 20, 20, 20]

    @pytest.mark.parametrize("ps, vec_or_num, expected_exception, message", [
        ([2, 3], [1], ValueError, r"Length of vec_or_num \(1\) does not match number of groups \(2\)"),
        ([2, 3], "not a number or list", TypeError, r"vec_or_num must be either a number"),
        ([2, 3], None, TypeError, r"vec_or_num must be either a number"),
    ])
    def test_group_expand_invalid(self, ps, vec_or_num, expected_exception, message):
        gf = GroupedFeatures(ps)
        with pytest.raises(expected_exception, match=message):
            gf.group_expand(vec_or_num)

    # Tests for fill

    @pytest.mark.parametrize("value, length, expected", [
        (5, 3, [5,5,5]),
        ("a", 2, ["a", "a"]),
        (None, 0, []),
    ])
    def test_fill_valid(self, value, length, expected):
        assert fill(value, length) == expected

    @pytest.mark.parametrize("value, length, expected_exception, message", [
        (5, -1, ValueError, "length must be a non-negative integer"),
        ("a", 2.5, TypeError, "length must be an integer"),
        (None, "3", TypeError, "length must be an integer"),
    ])
    def test_fill_invalid(self, value, length, expected_exception, message):
        with pytest.raises(expected_exception, match=message):
            fill(value, length)

# Additional tests for edge cases

class TestGroupedFeaturesEdgeCases:
    def test_empty_groups(self):
        ps = []
        gf = GroupedFeatures(ps)
        assert gf.ngroups() == 0
        assert gf.nfeatures() == 0
        # group_summary on empty groups
        summary = gf.group_summary([], sum_func)
        assert summary == []

    def test_single_group(self):
        ps = [5]
        gf = GroupedFeatures(ps)
        assert gf.ngroups() == 1
        assert gf.nfeatures() == 5
        assert list(gf.group_idx(0)) == list(range(0,5))
        expanded = gf.group_expand([10])
        assert expanded == [10,10,10,10,10]

    def test_group_expand_with_floats(self):
        ps = [1, 2]
        gf = GroupedFeatures(ps)
        vec_or_num = [3.5, 4.5]
        expanded = gf.group_expand(vec_or_num)
        assert expanded == [3.5, 4.5, 4.5]

    def test_group_summary_empty(self):
        ps = []
        gf = GroupedFeatures(ps)
        vec = []
        summary = gf.group_summary(vec, sum_func)
        assert summary == []

    def test_group_expand_empty(self):
        ps = []
        gf = GroupedFeatures(ps)
        expanded_num = gf.group_expand(5)
        expanded_list = gf.group_expand([])
        assert expanded_num == []
        assert expanded_list == []

    def test_group_summary_non_numeric(self):
        ps = [2, 2]
        gf = GroupedFeatures(ps)
        vec = ["a", "b", "c", "d"]
        def concat(group: List[str]) -> str:
            return ''.join(group)
        summary = gf.group_summary(vec, concat)
        assert summary == ["ab", "cd"]

    def test_group_expand_with_non_numeric_list(self):
        ps = [2, 3]
        gf = GroupedFeatures(ps)
        vec_or_num = ['x', 'y']
        expanded = gf.group_expand(vec_or_num)
        assert expanded == ['x', 'x', 'y', 'y', 'y']

    def test_group_summary_with_different_types(self):
        ps = [1, 1]
        gf = GroupedFeatures(ps)
        vec = [True, False]
        def any_func(group: List[bool]) -> bool:
            return any(group)
        summary = gf.group_summary(vec, any_func)
        assert summary == [True, False]

