import pytest
import numpy as np
from merge_groups import merge_groups

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(100, 50)  # 100 features, 50 samples

@pytest.fixture
def sample_groups():
    return {
        "group1": [0, 1, 2, 3, 4],
        "group2": [5, 6, 7, 8, 9],
        "group3": [10, 11, 12, 13, 14],
        "group4": [15, 16, 17, 18, 19]
    }

def test_normal_operation(sample_data, sample_groups):
    result = merge_groups(sample_data, sample_groups, 3)
    assert isinstance(result, dict)
    assert "new_groups" in result
    assert "new_group_members" in result
    assert len(result["new_groups"]) == 3
    assert len(result["new_group_members"]) == 3

def test_max_groups_edge_case(sample_data, sample_groups):
    result = merge_groups(sample_data, sample_groups, 4)
    assert len(result["new_groups"]) == 4
    assert len(result["new_group_members"]) == 4

def test_different_distance_method(sample_data, sample_groups):
    result = merge_groups(sample_data, sample_groups, 3, method_distance="euclidean")
    assert len(result["new_groups"]) == 3

def test_different_clustering_method(sample_data, sample_groups):
    result = merge_groups(sample_data, sample_groups, 3, method_clust="average")
    assert len(result["new_groups"]) == 3

def test_invalid_high_dim_data():
    with pytest.raises(ValueError):
        merge_groups(None, {"group1": [0, 1]}, 2)

def test_invalid_init_groups():
    with pytest.raises(ValueError):
        merge_groups(np.random.rand(10, 5), None, 2)

def test_invalid_max_groups():
    with pytest.raises(ValueError):
        merge_groups(np.random.rand(10, 5), {"group1": [0, 1]}, None)

def test_max_groups_too_low():
    with pytest.raises(ValueError):
        merge_groups(np.random.rand(10, 5), {"group1": [0, 1], "group2": [2, 3]}, 1)

def test_max_groups_too_high():
    with pytest.raises(ValueError):
        merge_groups(np.random.rand(10, 5), {"group1": [0, 1], "group2": [2, 3]}, 3)

def test_single_feature_groups(sample_data):
    single_feature_groups = {f"group{i}": [i] for i in range(10)}
    result = merge_groups(sample_data, single_feature_groups, 5)
    assert len(result["new_groups"]) == 5

def test_uneven_group_sizes(sample_data):
    uneven_groups = {
        "group1": [0],
        "group2": [1, 2, 3, 4, 5],
        "group3": [6, 7],
        "group4": [8, 9, 10, 11]
    }
    result = merge_groups(sample_data, uneven_groups, 3)
    assert len(result["new_groups"]) == 3

def test_all_features_in_one_group(sample_data):
    all_in_one = {"group1": list(range(sample_data.shape[0]))}
    with pytest.raises(ValueError):
        merge_groups(sample_data, all_in_one, 2)

def test_output_consistency(sample_data, sample_groups):
    result1 = merge_groups(sample_data, sample_groups, 3)
    result2 = merge_groups(sample_data, sample_groups, 3)
    assert result1 == result2

def test_large_dataset():
    large_data = np.random.rand(1000, 500)
    large_groups = {f"group{i}": list(range(i*10, (i+1)*10)) for i in range(100)}
    result = merge_groups(large_data, large_groups, 10)
    assert len(result["new_groups"]) == 10

if __name__ == "__main__":
    pytest.main()
