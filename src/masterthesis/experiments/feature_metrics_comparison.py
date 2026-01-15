#%% [markdown]
"""
    This file considers the comparison of different feature representation metrics.
    We compare the weight-based representation with the feature-based representation.
"""

#%%

import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from masterthesis.icfg import ICfg, Expr
from masterthesis.trainer import Trainer
from masterthesis.feature_generation import IndependentSparseGen
from masterthesis.toy_models_metrics import RepresentedFeatureMetric, RepresentedFeatureWeightMetric
from masterthesis.stacked_torch_modules import StackedCoupledLinearModel, WeightedMSELoss
from masterthesis import utils
from masterthesis.experiment_logger import ExpLogger
from masterthesis.utils import to_np, set_seed
from masterthesis.style_reference import *

test = False

prefix = "small"

n_repeats = 16
if prefix == "small":
    n_sparsities = 8
    n_features = 5
    n_hidden = 2
    importances = Expr(f'(0.9**th.arange({n_features}))[None, :]')
elif prefix == "large":
    n_sparsities = 8
    n_features = 100
    n_hidden = 10
    importances = Expr(f'(0.999**th.arange({n_features}))[None, :]') # for the large experiment, we use a more gradual importance decay
else:
    raise ValueError(f"Unknown prefix: {prefix}")

original_sparsity = (20 ** -th.linspace(0, 1, n_sparsities))[:, None]
repeated_feature_probability = original_sparsity.squeeze(1).repeat_interleave(n_repeats)[:, None]
n_instances = n_sparsities * n_repeats

icfg = ICfg(
        n_sparsities=n_sparsities,
        n_repeats=n_repeats,
        model=ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=n_instances,
            n_features=n_features,
            n_hidden=n_hidden,
            activation=nn.ReLU(),
            bias=True
        ),
        trainer=ICfg(
            cls=Trainer,
            n_iterations=10_000 if not test else 10,
            n_batch=1024, # new argument ; moved to trainer (previously passed to feature generator)
            optimizer_cls=optim.AdamW,
            optimizer_kwargs=ICfg(lr=1e-3, weight_decay=1e-2),
            feature_generator = ICfg(
                cls=IndependentSparseGen,  # Use the new independent sparse generator
                feature_probability=repeated_feature_probability,  
                distribution='uniform', 
            ),
            loss = ICfg(
                cls = WeightedMSELoss,
                importances=importances,  # Importance weights for features
            ),
        ),
        feat_metrics = [
            ("ε = 0.1 - represented", ICfg(cls=RepresentedFeatureMetric, epsilon=0.1, n_features = n_features, n_instances = n_instances)), 
            ("ε = 0.2 - represented", ICfg(cls=RepresentedFeatureMetric, epsilon=0.2, n_features = n_features, n_instances = n_instances)), 
            ("min norm = 0.4 - represented", ICfg(cls=RepresentedFeatureWeightMetric, min_norm=0.4)),
            ("min norm = 0.6 - represented", ICfg(cls=RepresentedFeatureWeightMetric, min_norm=0.6))],
    )

# train

logger = ExpLogger(
    base_name="feature_metrics_comparison",
    prefix=prefix)

set_seed()
ecfg = icfg.eval()

dev = utils.get_free_gpu()
ecfg.trainer.train(
    model=ecfg.model.to(dev),
)
logger.save_models_cfgs_new(models_dict={'model': ecfg.model}, cfgs_dict={'model': icfg})

#%% Collect Results
instance_idx = np.arange(ecfg.n_sparsities).repeat(ecfg.n_repeats)

# Debug: Check shapes
print(f"n_instances: {ecfg.model.n_instances}")
print(f"instance_idx shape: {instance_idx.shape}")
print(f"feature_probability shape: {ecfg.trainer.feature_generator.feature_probability.shape}")

df = pd.DataFrame({
    'Instance Idx': np.arange(ecfg.model.n_instances),
    'Sparsity Group': instance_idx,
    'Feature Probability': to_np(ecfg.trainer.feature_generator.feature_probability).flatten(),
})

for (metric_label, metric) in ecfg.feat_metrics:
    vals = metric(ecfg.model)
    df[metric_label] = to_np(vals)

# Plot: Feature Probability vs. Represented Features for each metric
melted = df.melt(id_vars=['Instance Idx', 'Sparsity Group', 'Feature Probability'], 
                 value_vars=[label for (label, _) in ecfg.feat_metrics],
                 var_name='Metric', value_name='Represented Features')

plt.figure(figsize=(8, 5))
p = sns.lineplot(data=melted, x='Feature Probability', y='Represented Features', hue='Metric', style='Metric', markers = True, **lineplot_kwargs, **errorbar_kwargs)
plt.gca().set(ylabel='Represented Features', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "feature_metrics_comparison",show=True)
