import pytest
from groupedfeatures import GroupedFeatures, fill

@pytest.fixture
def gf():
    return GroupedFeatures([2, 3, 4])

def test_initialization(gf):
    assert gf.ps == [2, 3, 4]
    assert gf.p == 9
    assert gf.num_groups == 3

def test_from_group_size():
    gf = GroupedFeatures.from_group_size(2, 3)
    assert gf.ps == [2, 2, 2]
    assert gf.p == 6
    assert gf.num_groups == 3

def test_ngroups(gf):
    assert gf.ngroups() == 3

def test_nfeatures(gf):
    assert gf.nfeatures() == 9

@pytest.mark.parametrize("index, expected", [
    (0, [0, 1]),
    (1, [2, 3, 4]),
    (2, [5, 6, 7, 8])
])
def test_group_idx(gf, index, expected):
    assert list(gf.group_idx(index)) == expected

def test_group_summary(gf):
    vec = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = gf.group_summary(vec, sum)
    assert result == [3, 12, 30]

def test_group_expand_list(gf):
    vec = [1, 2, 3]
    result = gf.group_expand(vec)
    assert result == [1, 1, 2, 2, 2, 3, 3, 3, 3]

def test_group_expand_scalar(gf):
    result = gf.group_expand(5)
    assert result == [5] * 9

def test_fill():
    assert fill(3, 5) == [3, 3, 3, 3, 3]

def test_invalid_group_idx(gf):
    with pytest.raises(IndexError):
        gf.group_idx(3)

def test_group_summary_empty():
    gf_empty = GroupedFeatures([])
    assert gf_empty.group_summary([], sum) == []

def test_group_expand_mismatch():
    gf = GroupedFeatures([2, 3])
    with pytest.raises(IndexError):
        gf.group_expand([1])  # Not enough elements in the input list