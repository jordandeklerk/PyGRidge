import pytest
import numpy as np
from create_partition import create_variable_groups

def test_numeric_input():
    vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = create_variable_groups(vec, num_groups=5, min_group_size=2)
    assert isinstance(result, dict)
    assert len(result) == 5
    assert all(isinstance(group, list) for group in result.values())

def test_string_input():
    vec = ['a', 'b', 'c', 'd', 'e']
    variable_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    result = create_variable_groups(vec, variable_names=variable_names)
    assert isinstance(result, dict)
    assert 'VarIn' in result and 'VarOut' in result
    assert set(result['VarIn']) == {0, 1, 2, 3, 4}
    assert set(result['VarOut']) == {5, 6}

def test_uniform_partition():
    vec = list(range(100))
    result = create_variable_groups(vec, uniform_groups=True, group_size=10)
    assert len(result) == 10
    assert all(len(group) == 10 for group in result.values())

def test_decreasing_order():
    vec = [5, 2, 8, 1, 9]
    result = create_variable_groups(vec, sort_descending=True, uniform_groups=True, group_size=1)
    expected_order = [4, 2, 0, 1, 3]
    assert list(result.values()) == [[i] for i in expected_order]

def test_increasing_order():
    vec = [5, 2, 8, 1, 9]
    result = create_variable_groups(vec, sort_descending=False, uniform_groups=True, group_size=1)
    expected_order = [3, 1, 0, 2, 4]
    assert list(result.values()) == [[i] for i in expected_order]

def test_subset():
    vec = [1, 2, 3, 4, 5]
    variable_names = ['a', 'b', 'c', 'd', 'e']
    subset = ['b', 'd', 'e']
    result = create_variable_groups(vec, variable_names=variable_names, subset=subset, uniform_groups=True, group_size=1)
    assert set(sum(result.values(), [])) == {1, 3, 4}

def test_non_uniform_partition():
    vec = list(range(100))
    result = create_variable_groups(vec, uniform_groups=False, num_groups=5, min_group_size=10)
    assert len(result) == 5
    assert all(len(group) >= 10 for group in result.values())

def test_empty_input():
    with pytest.raises(ValueError, match="Input vector is empty"):
        create_variable_groups([])

def test_single_element_input():
    result = create_variable_groups([1], uniform_groups=True, group_size=1)
    assert len(result) == 1
    assert list(result.values()) == [[0]]

def test_invalid_input_type():
    with pytest.raises(ValueError):
        create_variable_groups([1, 'a'])

def test_incompatible_parameters():
    vec = list(range(10))
    result = create_variable_groups(vec, num_groups=10, min_group_size=2)
    assert len(result) == 10
    assert all(len(group) == 1 for group in result.values())

def test_missing_variable_names():
    with pytest.raises(ValueError):
        create_variable_groups(['a', 'b', 'c'])

def test_numpy_array_input():
    vec = np.array([1, 2, 3, 4, 5])
    result = create_variable_groups(vec, num_groups=5, min_group_size=1)
    assert isinstance(result, dict)
    assert len(result) == 5

def test_large_input():
    vec = list(range(1000))
    result = create_variable_groups(vec, num_groups=20, min_group_size=10)
    assert len(result) == 20
    assert sum(len(group) for group in result.values()) == 1000

if __name__ == "__main__":
    pytest.main()
