import pytest
import numpy as np
from ..src.seagull_bisection import seagull_bisection


def test_seagull_bisection_basic():
    """Test basic functionality of seagull_bisection."""
    rows = 5
    alpha = 0.5
    left_border = 0.0
    right_border = 1.0
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    vector_in = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    result = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    assert 0.0 <= result <= 1.0


def test_seagull_bisection_edge_cases():
    """Test seagull_bisection with alpha at boundary values."""
    with pytest.raises(ValueError):
        seagull_bisection(5, 0.0, 0.0, 1.0, 1.0, np.ones(5), np.ones(5))

    with pytest.raises(ValueError):
        seagull_bisection(5, 1.0, 0.0, 1.0, 1.0, np.ones(5), np.ones(5))


def test_seagull_bisection_convergence():
    """Test convergence of seagull_bisection results."""
    rows = 10
    alpha = 0.5
    left_border = 0.0
    right_border = 10.0
    group_weight = 1.0
    vector_weights = np.ones(rows)
    vector_in = np.arange(rows) / rows

    result1 = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    result2 = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )

    assert np.isclose(result1, result2)


def test_seagull_bisection_negative_values():
    """Test seagull_bisection with negative input values."""
    rows = 5
    alpha = 0.5
    left_border = -1.0
    right_border = 1.0
    group_weight = 1.0
    vector_weights = np.ones(rows)
    vector_in = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])

    result = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    assert -1.0 <= result <= 1.0


def test_seagull_bisection_large_values():
    """Test seagull_bisection with large input values."""
    rows = 5
    alpha = 0.5
    left_border = 0.0
    right_border = 1e6
    group_weight = 1e3
    vector_weights = np.ones(rows) * 1e3
    vector_in = np.array([1e3, 1e4, 1e5, 1e6, 1e7])

    result = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    assert 0.0 <= result <= 1e6


def test_seagull_bisection_small_values():
    """Test seagull_bisection with small input values."""
    rows = 5
    alpha = 0.5
    left_border = 0.0
    right_border = 1e-6
    group_weight = 1e-3
    vector_weights = np.ones(rows) * 1e-3
    vector_in = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])

    result = seagull_bisection(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    assert 0.0 <= result <= 1e-6


def test_seagull_bisection_input_validation():
    """Test input validation for seagull_bisection."""
    rows = 5
    vector_weights = np.ones(rows)
    vector_in = np.ones(rows)

    with pytest.raises(ValueError):
        seagull_bisection(rows, -0.1, 0.0, 1.0, 1.0, vector_weights, vector_in)

    with pytest.raises(ValueError):
        seagull_bisection(rows, 1.1, 0.0, 1.0, 1.0, vector_weights, vector_in)

    with pytest.raises(ValueError):
        seagull_bisection(rows, 0.5, 1.0, 0.0, 1.0, vector_weights, vector_in)

    with pytest.raises(ValueError):
        seagull_bisection(rows, 0.5, 0.0, 1.0, 1.0, vector_weights[:-1], vector_in)


if __name__ == "__main__":
    pytest.main()
