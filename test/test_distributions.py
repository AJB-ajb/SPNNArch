import torch as th
import pytest
from collections import namedtuple
from masterthesis.utils import (
    generate_batch_independent_sparse_uniform,
    generate_batch_dependent_sparse_uniform
)

# Create a mock config object for testing
@pytest.fixture
def mock_config():
    Config = namedtuple('Config', ['device', 'n_batch', 'n_instances', 'n_features', 'feature_probability'])
    return Config(
        device=th.device('cpu'),
        n_batch=5,
        n_instances=3,
        n_features=10,
        feature_probability=0.3
    )

class TestIndependentSparseUniform:
    def test_shape_and_sparsity(self, mock_config):
        result = generate_batch_independent_sparse_uniform(mock_config)
        
        # Check shape
        assert result.shape == (mock_config.n_batch, mock_config.n_instances, mock_config.n_features)
        
        # Check device
        assert result.device == mock_config.device
        
        # Check sparsity
        non_zero_ratio = (result != 0).float().mean().item()
        # Allow for some statistical variance
        assert 0.15 < non_zero_ratio < 0.45  # feature_probability = 0.3
        
        # Check range of non-zero values
        non_zero_values = result[result != 0]
        if len(non_zero_values) > 0:  # Handle case where all values are 0
            assert (non_zero_values >= 0).all() and (non_zero_values <= 1).all()

class TestDependentSparseUniform:
    def test_shape_and_properties(self, mock_config):
        result = generate_batch_dependent_sparse_uniform(mock_config)
        
        # Check shape
        assert result.shape == (mock_config.n_batch, mock_config.n_instances, mock_config.n_features)
        
        # Check device
        assert result.device == mock_config.device
        
        # Check that each instance has exactly one active feature
        active_count = (result != 0).sum(dim=2)
        assert (active_count == 1).all()
        
        # Check range of non-zero values
        non_zero_values = result[result != 0]
        assert (non_zero_values >= 0).all() and (non_zero_values <= 1).all()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
