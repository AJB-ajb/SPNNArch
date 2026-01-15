import torch as th
from abc import ABC, abstractmethod
from .stacked_torch_modules import expand_instances

# ------ Feature generation ----

## Distribution: independent sparse × uniform

def draw(shape, distribution):
    # Uses current default device context
    if distribution == 'uniform':
        return th.rand(shape)
    if distribution == 'Gaussian':
        return th.randn(shape)
    if distribution == 'ones': # for boolean features
        return th.ones(shape)
    raise NotImplementedError()

# For generators that need additional parameters, we can use a class-based approach
# For generators without additional parameters, we can use simple functions

class FeatureGenerator(ABC):
    """
    Abstract base class for feature generators.
    """
    @abstractmethod
    def __call__(self, n_batch, n_features, n_instances, device=None):
        """
        Generate a batch of features.
        
        Args:
            n_batch: Number of samples in the batch
            n_features: Number of features in each sample
            n_instances: Number of instances (models) to generate data for
            device: Device to generate tensors on
        """
        pass

class IndependentSparseGen(FeatureGenerator):
    """
    Generator for independent sparse uniform features.
    Each feature is active with a given probability and drawn from a specified distribution.
    """
    def __init__(self, feature_probability, distribution='uniform'):
        self.feature_probability = feature_probability
        self.distribution = distribution

    def __call__(self, n_batch, n_features, n_instances, device=None):
        feature_probability = th.tensor(self.feature_probability, device=device)
        shape = (n_batch, 1, n_features)
        mask = th.rand(shape, device=device).expand(-1, n_instances, -1) <= feature_probability

        base_distribution = draw(shape, self.distribution).to(device)
        batch = th.where(mask, base_distribution, th.zeros(shape, device=device))
        return batch

class DependentSparseGen(FeatureGenerator):
    """
    Generator for dependent sparse uniform features.
    Each instance has exactly one active feature, drawn from a specified distribution.
    """
    def __init__(self, distribution='uniform'):
        self.distribution = distribution
    def __call__(self, n_batch, n_features, n_instances, device=None):
        shape = (n_batch, n_features)
        with th.device(device):
            batch = th.zeros(shape)
            index_active = th.randint(0, n_features, (n_batch,))
            values = draw((n_batch,), self.distribution).to(device)
            
            # Set values at the active indices
            batch[th.arange(n_batch), index_active] = values
        return expand_instances(batch, n_instances)

def generate_ones_regular(n_batch, n_features, n_instances, device=None):
    """
    Generate an eye matrix of shape (n_batch, n_features), i.e. all features are active for each sample.
    
    Args:
        n_batch: Number of samples in the batch (not used here)
        n_features: Number of features (dimensions)
        n_instances: Number of instances (models) to generate data for
        device: Device to generate tensors on
    """
    return expand_instances(th.eye(n_batch, n_features, device=device), n_instances)