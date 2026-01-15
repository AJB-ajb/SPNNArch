import torch as th
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.linalg as linalg
from .config import Cfg, replace
from .feature_generation import draw



base_project_dir = Path(__file__).resolve().parents[2]
experiments_dir = base_project_dir / "experiments"

def get_free_gpu():
    """Select the GPU with the most available memory ; Fallback to CPU if no GPU is available."""
    if not th.cuda.is_available():
        return th.device('cpu')
    
    max_free_memory = 0
    best_device = 0
    
    for i in range(th.cuda.device_count()):
        th.cuda.set_device(i)
        free_memory = th.cuda.get_device_properties(i).total_memory - th.cuda.memory_allocated(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i
    
    return th.device(f'cuda:{best_device}')

def set_seed(seed = 42):
    """
    Set the random seed for reproducibility.
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

def to_np(val):
    if type(val) == th.Tensor:
        return val.detach().cpu().numpy()
    else:
        return np.array(val)

# function to print "mean ± σ [min, max]" for a tensor
def print_stats(x, desc = ""):
    x = th.tensor(x)
    mean = x.mean().item()
    print(f"{desc} {mean: .3e} ± {x.std().item() / mean :.2%} [{x.min().item():.2e}, {x.max().item():.2e}]")


# --------- Initialization utilities ---------

def get_functional_name(activation):
    """
    Map activation function instances/classes to functional names for nn.init.calculate_gain().
    
    This is the standard PyTorch way - using a simple mapping from class names
    to the string names that calculate_gain() expects.
    
    Args:
        activation: Either an activation function instance (e.g., nn.ReLU()) or class (e.g., nn.ReLU)
        
    Returns:
        str: The functional name expected by nn.init.calculate_gain()
    """
    # Handle both instances and classes
    if hasattr(activation, '__class__'):
        class_name = activation.__class__.__name__
    else:
        class_name = activation.__name__
    
    # Standard mapping from PyTorch class names to functional names
    mapping = {
        'ReLU': 'relu',
        'LeakyReLU': 'leaky_relu',  
        'Tanh': 'tanh',
        'Sigmoid': 'sigmoid',
        'Identity': 'linear',
        'Linear': 'linear',
    }
    
    return mapping.get(class_name, 'linear')  # Default to 'linear'

# --------- Training ---------
def linear_warmup(step, max_steps):
    return min((step + 1) / max_steps, 1.0)
def cosine_annealing(step, max_steps):
    return 0.5 * (1 + np.cos(np.pi * step / max_steps))

def cosine_annealing_warmup(optimizer, warmup_steps, total_steps):
    def lr_scheduler(step):
        if step < warmup_steps:
            return linear_warmup(step, warmup_steps)
        else:
            return cosine_annealing(step - warmup_steps, total_steps - warmup_steps)
    return th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

# ------ Feature generation ----
# ---- All deprecated ----

def batched_feature_generator(fn):
    """Wrap a batch generation function to an infinite generator."""
    def generator(cfg):
        while True:
            yield fn(cfg)
    return generator

def batched_feature_generator_kwargs(fn):
    """Wrap a batch generation function to an infinite generator that accepts kwargs."""
    def generator(**kwargs):
        while True:
            yield fn(**kwargs)
    return generator


from deprecated import deprecated

@deprecated(reason="Use IndependentSparseUniformGen instead")        
def generate_batch_independent_sparse_uniform(cfg):
    """
        Generate a batch of sparse feature data.
        Each feature is active with probability `cfg.feature_probability`. If active, the value is drawn from a distribution specified by cfg.
    """
    # Handle both Cfg objects with keys() method and simple objects/namedtuples
    if hasattr(cfg, 'keys') and callable(cfg.keys):
        base_distribution = 'uniform' if 'distribution' not in cfg.keys() else cfg.distribution
    else:
        base_distribution = 'uniform' if not hasattr(cfg, 'distribution') else cfg.distribution

    shape = (cfg.n_batch,1,cfg.n_features)
    mask = th.rand(shape, device=cfg.device).expand(-1, cfg.n_instances, -1) <= cfg.feature_probability

    base_distribution = draw(shape, base_distribution).to(cfg.device)
    batch = th.where(mask, base_distribution, th.zeros(shape, device=cfg.device))
    return batch.to(cfg.device)

def generate_batch_independent_sparse_uniform_kwargs(
    n_batch, n_features, n_instances, feature_probability, device, distribution='uniform'
):
    """
    Generate a batch of sparse feature data using direct parameters.
    Each feature is active with probability `feature_probability`. If active, the value is drawn from the specified distribution.
    
    Args:
        n_batch: Number of samples in the batch
        n_features: Number of features in each sample
        n_instances: Number of instances (models) to generate data for
        feature_probability: Probability tensor for each feature to be active
        device: Device to generate tensors on
        distribution: Distribution to draw values from ('uniform', 'Gaussian', 'ones')
    """
    with th.device(device):
        # Ensure feature_probability is on the correct device
        feature_probability = feature_probability.to(device)
        
        shape = (n_batch, 1, n_features)
        mask = th.rand(shape).expand(-1, n_instances, -1) <= feature_probability

        base_distribution = draw(shape, distribution)
        batch = th.where(mask, base_distribution, th.zeros(shape))
        return batch

## Distribution: dependent sparse × uniform : only one feature is active at a time
def generate_batch_dependent_sparse_uniform(cfg):
    """
        Generate a batch of sparse feature data.
        Each instance has exactly one active feature, drawn from a uniform distribution, i.e. feature probability is unused here and not meaningful.
    """
    shape = (cfg.n_batch, cfg.n_features)
    with th.device(cfg.device):
        batch = th.zeros(shape)
        index_active = th.randint(0, cfg.n_features, (cfg.n_batch,))
        values = th.rand(cfg.n_batch)
        
        # Set values at the active indices
        batch[th.arange(cfg.n_batch), index_active] = values
    
    # Expand to include instances dimension: (n_batch, n_features) -> (n_batch, n_instances, n_features)
    # Note: All instances get the same feature pattern
    return batch.unsqueeze(1).expand(-1, cfg.n_instances, -1)

def generate_batch_dependent_sparse_uniform_kwargs(n_batch, n_features, device = None, **kwargs):
    """
    Generate a batch of sparse feature data using direct parameters.
    Each instance has exactly one active feature, drawn from a uniform distribution.
    
    Args:
        n_batch: Number of samples in the batch
        n_features: Number of features in each sample
        device: Device to generate tensors on
        **kwargs: Additional arguments (ignored for this generator)
    """
    shape = (n_batch, n_features)
    with th.device(device):
        batch = th.zeros(shape)
        index_active = th.randint(0, n_features, (n_batch,))
        values = th.rand(n_batch)
        
        # Set values at the active indices
        batch[th.arange(n_batch), index_active] = values
    return batch

def get_feature_generator(generator_type: str, use_kwargs=False):
    """Get the appropriate feature generator function based on string type.
    
    Args:
        generator_type: Type of generator ('sparse_uniform', 'dependent_sparse', etc.)
        use_kwargs: If True, return kwargs-based generator functions
    """
    if use_kwargs:
        generators = {
            "sparse_uniform": generate_batch_independent_sparse_uniform_kwargs,
            "dependent_sparse": generate_batch_dependent_sparse_uniform_kwargs,
            "independent_sparse": generate_batch_independent_sparse_uniform_kwargs,  # alias
            "ones_feature": ones_feature_kwargs,
            "ones_regular": ones_regular_kwargs,
        }
    else:
        generators = {
            "sparse_uniform": generate_batch_independent_sparse_uniform,
            "dependent_sparse": generate_batch_dependent_sparse_uniform,
            "independent_sparse": generate_batch_independent_sparse_uniform,  # alias
            "ones_feature": ones_feature,
            "ones_regular": ones_regular,
        }
    
    if generator_type not in generators:
        available = ", ".join(generators.keys())
        raise ValueError(f"Unknown feature generator type: {generator_type}. Available: {available}")
    
    return generators[generator_type]

def instantiate_feature_generator(generator_type: str, use_kwargs=False):
    """ Instantiate a feature generator function that yields batches indefinitely. 
    
    Args:
        generator_type: Type of generator to instantiate
        use_kwargs: If True, return a generator that accepts kwargs instead of cfg
    """
    if use_kwargs:
        return batched_feature_generator_kwargs(get_feature_generator(generator_type, use_kwargs=True))
    else:
        return batched_feature_generator(get_feature_generator(generator_type, use_kwargs=False))

def ones_feature(cfg):
    """
        Return a tensor of shape (cfg.n_batch,cfg.n_features) where exactly one feature is active per sample, with value 1.
    """
    with th.device(cfg.device):
        index_active = th.randint(0, cfg.n_features, (cfg.n_batch,))
        batch = th.zeros((cfg.n_batch, cfg.n_features))
        batch[th.arange(cfg.n_batch), index_active] = 1
    return batch

def ones_feature_kwargs(n_batch, n_features, device, **kwargs):
    """
    Return a tensor of shape (n_batch, n_features) where exactly one feature is active per sample, with value 1.
    
    Args:
        n_batch: Number of samples in the batch
        n_features: Number of features in each sample
        device: Device to generate tensors on
        **kwargs: Additional arguments (ignored)
    """
    with th.device(device):
        index_active = th.randint(0, n_features, (n_batch,))
        
        batch = th.zeros((n_batch, n_features))
        batch[th.arange(n_batch), index_active] = 1
    return batch

def ones_regular(cfg):
    return th.eye(cfg.n_features, device=cfg.device)

def ones_regular_kwargs(n_features, device, **kwargs):
    """
    Return an identity matrix of shape (n_features, n_features).
    
    Args:
        n_features: Number of features (dimensions)
        device: Device to generate tensors on
        **kwargs: Additional arguments (ignored)
    """
    return th.eye(n_features, device=device)

