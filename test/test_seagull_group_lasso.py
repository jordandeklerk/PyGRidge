import numpy as np
import pytest
from ..src.seagull_group_lasso import seagull_group_lasso


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, p = 100, 20
    y = np.random.randn(n)
    X = np.random.randn(n, p)
    feature_weights = np.ones(p)
    groups = np.repeat(np.arange(1, 6), 4)
    beta = np.zeros(p)
    index_permutation = np.arange(1, p + 1)
    return {
        "y": y,
        "X": X,
        "feature_weights": feature_weights,
        "groups": groups,
        "beta": beta,
        "index_permutation": index_permutation,
        "epsilon_convergence": 1e-6,
        "max_iterations": 1000,
        "gamma": 0.9,
        "lambda_max": 1.0,
        "proportion_xi": 0.01,
        "num_intervals": 10,
        "num_fixed_effects": 0,
        "trace_progress": False,
    }


def test_seagull_group_lasso_basic(sample_data):
    result = seagull_group_lasso(**sample_data)
    assert isinstance(result, dict)
    assert "random_effects" in result
    assert "lambda" in result
    assert "iterations" in result


def test_seagull_group_lasso_dimensions(sample_data):
    result = seagull_group_lasso(**sample_data)
    n, p = sample_data["X"].shape
    num_intervals = sample_data["num_intervals"]
    assert result["random_effects"].shape == (num_intervals, p)
    assert result["lambda"].shape == (num_intervals,)
    assert result["iterations"].shape == (num_intervals,)


def test_seagull_group_lasso_fixed_effects(sample_data):
    sample_data["num_fixed_effects"] = 5
    result = seagull_group_lasso(**sample_data)
    assert "fixed_effects" in result
    assert "random_effects" in result
    assert result["fixed_effects"].shape == (sample_data["num_intervals"], 5)
    assert result["random_effects"].shape == (sample_data["num_intervals"], 15)


def test_seagull_group_lasso_convergence(sample_data):
    result = seagull_group_lasso(**sample_data)
    assert np.all(result["iterations"] <= sample_data["max_iterations"])


def test_seagull_group_lasso_lambda_range(sample_data):
    result = seagull_group_lasso(**sample_data)
    assert np.all(result["lambda"] <= sample_data["lambda_max"])
    assert np.all(
        result["lambda"] >= sample_data["lambda_max"] * sample_data["proportion_xi"]
    )


def test_seagull_group_lasso_trace_progress(sample_data, capsys):
    sample_data["trace_progress"] = True
    seagull_group_lasso(**sample_data)
    captured = capsys.readouterr()
    assert "Loop: 10 of 10 finished" in captured.out


def test_seagull_group_lasso_input_validation(sample_data):
    with pytest.raises(ValueError):
        invalid_data = sample_data.copy()
        invalid_data["X"] = invalid_data["X"][:, :-1]  # Mismatch dimensions
        seagull_group_lasso(**invalid_data)


if __name__ == "__main__":
    pytest.main()
