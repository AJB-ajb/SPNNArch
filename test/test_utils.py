import pytest
import torch as th

import math

from masterthesis.toy_models_metrics import feature_dimensions, feature_dimensions_2

# test calculation of feature dimension

class TestCalcUtils:
    def test_feature_dimensions(self):
        W = th.tensor([
            [[1., 0], [0, 1]],
            [[1, 0], [0, 1]]
        ])
        expected_dims = th.tensor([1.0, 1.0])
        calculated_dims = feature_dimensions(W)
        assert th.allclose(calculated_dims, expected_dims), f"Expected {expected_dims}, but got {calculated_dims}"

        for scaling in [0.1, 1, 10]:
            assert th.allclose(
                feature_dimensions(W * scaling),
                expected_dims), f"Expected {expected_dims}, but got {feature_dimensions(W * scaling)}"
            
        # test with cyclic dimensions, i.e. n_features = n, n_hidden = 2, and W_i = cos(i * 2π / n_features)

        n_features = 7
        W = th.tensor([[
            [math.cos(i * 2 * th.pi / n_features), math.sin(i * 2 * th.pi / n_features)] for i in range(n_features)]
        ]) # 1 × n_features × 2
        print(W.shape)

        expected_dims = 2 / n_features * th.ones(n_features)
        calculated_dims = feature_dimensions(W)
        assert th.allclose(calculated_dims, expected_dims), f"Expected {expected_dims}, but got {calculated_dims}"

        # test with vectors of different lengths (?)
        W = th.tensor([
            [[1., 0], [-0.5, 0.]],
        ])
        # the feature dimension her is 1 / (1 + 0.5^2) = 1 / 1.25 = 0.8
        expected_dims = th.tensor([[0.8, 0.2]])
        calculated_dims = feature_dimensions(W)
        assert th.allclose(calculated_dims, expected_dims), f"Expected {expected_dims}, but got {calculated_dims}"

    def test_feature_dimensions_2(self):
        # Test for a simple case
        W = th.tensor([
            [1., 0],
            [0, 1]
        ])
        # For orthonormal vectors, each feature dimension should be 1
        expected_dims = th.tensor([1.0, 1.0])
        calculated_dims = th.stack([feature_dimensions_2(W, i) for i in range(W.shape[0])])
        assert th.allclose(calculated_dims, expected_dims), f"Expected {expected_dims}, but got {calculated_dims}"

        # Test for non-orthogonal vectors
        W = th.tensor([
            [1., 0],
            [-0.5, 0.]
        ])
        # The feature dimension here is 1 / (1 + 0.5^2) = 0.8 for the first, 0.2 for the second
        expected_dims = th.tensor([0.8, 0.2])
        calculated_dims = th.stack([feature_dimensions_2(W, i) for i in range(W.shape[0])])
        assert th.allclose(calculated_dims, expected_dims), f"Expected {expected_dims}, but got {calculated_dims}"

    def test_feature_dimensions_agreement(self):
        # Test that feature_dimensions and feature_dimensions_2 agree for a batch of 1
        W = th.tensor([
            [[1., 0], [0, 1]],
            [[1., 0], [-0.5, 0.]]
        ])  # shape (2, 2, 2)
        for i in range(W.shape[0]):
            dims_2 = th.stack([feature_dimensions_2(W[i], j) for j in range(W.shape[1])])
            dims = feature_dimensions(W[i:i+1])[0]
            assert th.allclose(dims, dims_2), f"feature_dimensions and feature_dimensions_2 disagree: {dims} vs {dims_2}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
