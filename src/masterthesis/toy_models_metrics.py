"""
Toy Models Metrics Module

This module contains various metrics for analyzing toy models of superposition.
It consolidates metrics that were previously scattered across different files,
including:
- Feature representation counting
- Feature dimensions calculation
- Polysemanticity and interference metrics
- Geometry and dimensionality metrics
- Loss-based metrics

All metrics assume the standard toy model setup with weight tensors of shape
(n_instances, n_features, n_hidden) and configurations from the Cfg class.
"""

import torch as th
import torch.nn.functional as F
from torch import linalg
import numpy as np
from typing import Optional, Tuple, Union
from .feature_generation import generate_ones_regular
from .stacked_torch_modules import *


# ============================================================================
# Feature Representation Metrics
# ============================================================================

def boolean_approx_loss(model, cfg):
    """
    Return the mean squared error between the model output and the identity matrix, averaged over features.
    
    This computes how well the model approximates the identity function for one-hot encoded inputs.
    
    Args:
        model: Trained toy model with forward method
        cfg: Configuration object with n_features and n_instances
        
    Returns:
        Tensor of shape (n_features, n_instances), where each entry corresponds to the 
        mean squared error for the corresponding feature and instance.
    """
    data = th.eye(cfg.n_features)[:, None, :].expand(-1, cfg.n_instances, -1)  # (n_features × n_instances × n_features)
    # model(data) is (n_features × n_instances × n_features)
    losses = (data - model(data)).pow(2).mean(dim=2)  # mean over features
    return losses


def count_represented_features(model, cfg, epsilon: float = 0.1):
    """
    Count how many features are represented in the model via approximation loss.
    
    A feature is considered represented if the representation error (boolean approximation loss) 
    is below the given epsilon threshold.
    
    Args:
        model: Trained toy model
        cfg: Configuration object
        epsilon: Threshold for representation error
        
    Returns:
        Tensor of length n_instances, where each entry is the number of represented 
        features for the corresponding instance.
    """
    losses = boolean_approx_loss(model, cfg)
    represented_features = (losses < epsilon).sum(dim=0)  # sum over features
    return represented_features


def count_represented_features_via_weight(model, cfg, min_norm: float = 0.4):
    """
    Count how many features are represented in the model via weight norm.
    
    A feature is considered represented if its weight vector has norm above the given threshold.
    
    Args:
        model: Trained toy model with layers['L_down'] containing weights
        cfg: Configuration object  
        min_norm: Minimum weight norm threshold for feature representation
        
    Returns:
        Tensor of length n_instances, where each entry is the number of represented 
        features for the corresponding instance.
    """
    W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
    norms = th.linalg.vector_norm(W, dim=2)  # (n_instances, n_features)
    represented_features = (norms > min_norm).sum(dim=1)  # sum over features
    return represented_features


def boolean_approx_loss_kwargs(model, n_features, n_instances, device=None): #! substitute with simpler version using the more modular feature generators
    """
    Calculate the boolean approximation loss for all features using kwargs.
    
    This computes how well the model approximates the identity function for one-hot encoded inputs.
    
    Args:
        model: Trained toy model with forward method
        n_features: Number of features
        n_instances: Number of instances (models)
        device: Device to create tensors on (optional, inferred from model if not provided)
        
    Returns:
        Tensor of shape (n_features, n_instances), where each entry corresponds to the 
        mean squared error for the corresponding feature and instance.
    """
    if device is None:
        device = next(model.parameters()).device
    
    data = th.eye(n_features, device=device)[:, None, :].expand(-1, n_instances, -1)  # (n_features × n_instances × n_features)
    # model(data) is (n_features × n_instances × n_features)
    losses = (data - model(data)).pow(2).mean(dim=2)  # mean over features
    return losses


def count_represented_features_kwargs(model, n_features, n_instances, epsilon=0.1, device=None):
    """
    Count how many features are represented in the model via approximation loss using kwargs.
    
    A feature is considered represented if the representation error (boolean approximation loss) 
    is below the given epsilon threshold.
    
    Args:
        model: Trained toy model
        n_features: Number of features
        n_instances: Number of instances (models)
        epsilon: Threshold for representation error
        device: Device to create tensors on (optional, inferred from model if not provided)
        
    Returns:
        Tensor of length n_instances, where each entry is the number of represented 
        features for the corresponding instance.
    """
    losses = boolean_approx_loss_kwargs(model, n_features, n_instances, device)
    represented_features = (losses < epsilon).sum(dim=0)  # sum over features
    return represented_features


def count_represented_features_via_weight_kwargs(model, min_norm=0.4):
    """
    Count how many features are represented in the model via weight norm using kwargs.
    
    A feature is considered represented if its weight vector has norm above the given threshold.
    
    Args:
        model: Trained toy model with layers['L_down'] containing weights
        min_norm: Minimum weight norm threshold for feature representation
        
    Returns:
        Tensor of length n_instances, where each entry is the number of represented 
        features for the corresponding instance.
    """
    W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
    norms = th.linalg.vector_norm(W, dim=2)  # (n_instances, n_features)
    represented_features = (norms > min_norm).sum(dim=1)  # sum over features
    return represented_features


class RepresentedFeatureMetric:
    """
    Metric to compute represented features based on a threshold epsilon.
    It computes the non-weighted error for each feature in each instance,
    given perfectly homogeneous Boolean features (identity matrix).
    A feature is considered represented if the error is below epsilon.
    """
    def __init__(self, epsilon, n_features=None, n_instances=None):
        self.epsilon = epsilon
        self.n_features = n_features
        self.n_instances = n_instances

    def __call__(self, model, n_features=None, n_instances=None):
        _dev = next(model.parameters()).device
        # compute non-importance weighted error for each feature in each instance, given perfectly homogeneous Boolean features (identity matrix)
        # a feature is represented if the error is below epsilon
        n_features = self.n_features or n_features
        n_instances = self.n_instances or n_instances
        data = generate_ones_regular(
            n_batch=n_features,
            n_features=self.n_features,
            n_instances=self.n_instances,
            device=_dev
        )

        losses = (data - model(data)).pow(2).max(dim=2).values # max error over features
        represented_features = (losses < self.epsilon).sum(dim=0)
        return represented_features

class RepresentedFeatureWeightMetric: # currently only for coupled linear models
    def __init__(self, min_norm):
        self.min_norm = min_norm
    def __call__(self, model):
        # compute the represented features via weight norm
        # a feature is represented if the norm of the corresponding entry in the weight matrix is below min_norm
        
        if type(model) == StackedCoupledLinearModel or (type(model) == StackedGatingModel and model.coupled):
            W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
            norms = th.linalg.vector_norm(W, dim=2)  # (n_instances, n_features)
            represented_features = (norms > self.min_norm).sum(dim=1)  # sum over features
        elif type(model) in (StackedLinearModel, StackedGatingModel): # count the number of features where sqrt(norm(W_i) * norm(V_i)) > min_norm
            W_down = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
            W_up = model.layers['L_up'].weight.detach()  # (n_instances, n_hidden, n_features)
            norms_down = th.linalg.vector_norm(W_down, dim=2)
            norms_up = th.linalg.vector_norm(W_up, dim=1)
            norms = th.sqrt(norms_down * norms_up)
            represented_features = (norms > self.min_norm).sum(dim=1)
        else:
            raise ValueError("Unsupported model type for RepresentedFeatureWeightMetric")
        
        return represented_features


# ============================================================================
# Feature Dimensions
# ============================================================================

def feature_dimensions_metric(model):
    """
    Metric to compute feature dimensions for each feature in each instance.
    
    The feature dimension is defined as:
        d_k = ||W_k||^2 / sum_j (dot(W_k / ||W_k||, W_j)^2)
    ,i.e. , it measures how much of its own dimension a feature occupies relative to how much from of that dimension is occupied by other features.
    
    """
    W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
    dims = feature_dimensions(W)  # (n_instances, n_features)
    return dims

    # generalized feature dimension for uncoupled model:
    # where W := W_down ; V := W_up
    # d_k = W_k ⋅ V_k / sum_j (hat(W_k) ⋅ V_j)^2
    

def feature_dimensions_2(W, i_f):
    """
    Compute feature dimension for a single feature in a single instance.
    
    This is the original implementation without the instance dimension.
    
    Args:
        W: Weight tensor of shape (n_features, n_hidden)
        i_f: Feature index
        
    Returns:
        Scalar tensor with the feature dimension for feature i_f
    """
    W_i = W[i_f]
    W_i_hat = W_i / linalg.vector_norm(W_i)
    return linalg.vector_norm(W_i)**2 / th.sum((W_i_hat @ W.transpose(0, 1))**2)


def feature_dimensions(W):
    """
    Compute feature dimensions for a weight tensor of shape (n_instances, n_features, n_hidden).
    
    For each instance and each feature k:
        d_k = ||W_k||^2 / sum_j (dot(W_k / ||W_k||, W_j)^2)
        
    This metric measures how much "space" each feature takes up, considering interference
    from other features.
    
    Args:
        W: Weight tensor of shape (n_instances, n_features, n_hidden)
        
    Returns:
        Tensor of shape (n_instances, n_features) with feature dimensions
    """
    # W: (B, F, H)
    norms = linalg.vector_norm(W, dim=-1)                     # (B, F)
    W_hat = W / norms[..., None]                              # (B, F, H)
    dot_matrix = th.bmm(W_hat, W.transpose(1, 2))             # (B, F, F)
    denominator = (dot_matrix ** 2).sum(dim=-1)               # (B, F)
    dims = norms ** 2 / denominator
    return dims


# ============================================================================
# Geometry and Dimensionality Metrics
# ============================================================================

def compute_geometry_metric(model, cfg):
    """
    Calculate the geometry metric m/||W||_F^2 for each instance.
    
    This metric from the toy models paper measures the relationship between
    the hidden dimension and the Frobenius norm of the weight matrix.
    
    Args:
        model: Trained toy model with layers['L_down'] containing weights
        cfg: Configuration object with n_hidden
        
    Returns:
        Tensor of shape (n_instances,) with the metric for each instance
    """
    W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
    frobenius_norms_squared = th.linalg.matrix_norm(W, 'fro', dim=(1, 2))**2  # (n_instances,)
    metric = cfg.n_hidden / frobenius_norms_squared
    return metric


def compute_dimensionality_fractions(W):
    """
    Compute dimensionality fractions as in the original toy models paper.
    
    This computes the effective dimensionality of each feature considering interference.
    
    Args:
        W: Weight tensor of shape (n_instances, n_features, n_hidden)
        
    Returns:
        Tensor of shape (n_instances, n_features) with dimensionality fractions
    """
    norms = th.linalg.norm(W, 2, dim=-1)
    W_unit = W / th.clamp(norms[:, :, None], 1e-6, float('inf'))
    
    interferences = (th.einsum('eah,ebh->eab', W_unit, W)**2).sum(-1)
    
    dim_fracs = (norms**2 / interferences)
    return dim_fracs


# ============================================================================
# Loss and Error Metrics
# ============================================================================

def total_reconstruction_loss(model, cfg, average: bool = False):
    """
    Calculate the total reconstruction loss for the model.
    
    Args:
        model: Trained toy model
        cfg: Configuration object
        average: If True, average the loss by the number of features per instance
        
    Returns:
        Tensor of shape (n_instances,) with total losses
    """
    # Generate identity batch for all features
    batch = th.eye(cfg.n_features)[:, None, :].expand(-1, cfg.n_instances, -1)
    total_diff = (model(batch) - batch).pow(2)
    losses = total_diff.sum(dim=(0, 2))  # sum over batch and features
    
    if average:
        losses = losses / cfg.n_features
        
    return losses


def feature_reconstruction_errors(model, cfg):
    """
    Calculate reconstruction error for each individual feature.
    
    Args:
        model: Trained toy model
        cfg: Configuration object
        
    Returns:
        Tensor of shape (n_features, n_instances) with reconstruction errors
    """
    data = th.eye(cfg.n_features)[:, None, :].expand(-1, cfg.n_instances, -1)
    errors = (model(data) - data).pow(2).mean(dim=2)  # mean over feature dimension
    return errors






