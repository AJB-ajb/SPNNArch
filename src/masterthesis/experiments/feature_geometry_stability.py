#%% Feature Geometry Stability Experiment using New Framework
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from masterthesis.utils import *
from masterthesis.stacked_torch_modules import *
from masterthesis.experiment_logger import ExpLogger
from masterthesis.icfg import ICfg, Expr
from masterthesis.trainer import Trainer
from masterthesis.feature_generation import IndependentSparseGen
from masterthesis.style_reference import lineplot_kwargs, errorbar_kwargs

def calculate_geometry_metric(model, n_hidden):
    """Calculate the geometry metric nf/||W||_F^2 for each instance."""
    W = model.layers['L_down'].weight.detach()  # (n_instances, n_features, n_hidden)
    frobenius_norms_squared = th.linalg.matrix_norm(W, 'fro', dim=(1, 2))**2  # (n_instances,)
    return n_hidden / frobenius_norms_squared

def geometry_stability_cfg(test=False):
    """Configuration for geometry stability experiment."""
    n_sparsities = 20
    n_repeats = 8
    feature_probabilities = (20 ** -th.linspace(0, 1, n_sparsities))
    repeated_feature_probability = feature_probabilities.repeat_interleave(n_repeats)[:, None]
    
    return ICfg(
        base_name="feature_geometry_stability",
        prefix="run",
        tmp=test,
        test=test,
        model=ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=n_sparsities * n_repeats,
            n_features=200,
            n_hidden=20,
            activation=nn.ReLU(),
            bias=True
        ),
        trainer=ICfg(
            cls=Trainer,
            n_iterations=100 if test else 10_000,
            n_batch=1024,
            optimizer_cls=optim.AdamW,
            optimizer_kwargs=ICfg(lr=1e-3, weight_decay=1e-2),
            feature_generator=ICfg(
                cls=IndependentSparseGen,
                feature_probability=repeated_feature_probability,
                distribution='uniform',
            ),
            loss=ICfg(
                cls=th.nn.MSELoss,
            ),
        ),
        # Store for analysis
        original_sparsity=feature_probabilities,
        n_sparsities=n_sparsities,
        n_repeats=n_repeats
    )

def train_geometry_model(cfg):
    """Train a single geometry model."""
    eval_cfg = cfg.eval()
    
    device = get_free_gpu()
    model = eval_cfg.model.to(device)
    trainer = eval_cfg.trainer
    
    trainer.train(model)
    return model

#%%
LOAD = True

base_cfg = geometry_stability_cfg(test=False)

if not LOAD:
    logger = ExpLogger(
        base_name=base_cfg.base_name,
        tmp=base_cfg.tmp,
        prefix=base_cfg.prefix,
    )
    
    utils.set_seed()
    model = train_geometry_model(base_cfg)
    
    # Save model and config
    logger.save_models_cfgs_new({"geometry_model": model}, {"geometry_model": base_cfg})
else:
    logger = ExpLogger.load_logger(base_cfg.base_name, prefix=base_cfg.prefix)

# Load results and compute metrics
models_dict, cfgs_dict = logger.load_models_cfgs_new()
model = list(models_dict.values())[0]
cfg = list(cfgs_dict.values())[0]

eval_cfg = cfg.eval()

# Calculate geometry metrics
geometry_metrics = calculate_geometry_metric(model, eval_cfg.model.n_hidden)
geometry_metrics_reshaped = geometry_metrics.reshape(cfg.n_sparsities, cfg.n_repeats)

# Create DataFrame for analysis
instance_idx = np.arange(cfg.n_sparsities * cfg.n_repeats) % cfg.n_repeats

feature_probability = to_np(eval_cfg.trainer.feature_generator.feature_probability).flatten()
x_values = 1 / cfg.original_sparsity.repeat_interleave(cfg.n_repeats)

df = pd.DataFrame({
    'Instance Index': instance_idx,
    'Feature Probability': feature_probability,
    '1/(1-S)': to_np(x_values),
    'Geometry Metric': to_np(geometry_metrics)
})

# Save results
df.to_csv(logger.path("geometry_stability_results.csv"), index=False)

#%% Create plots using seaborn

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='1/(1-S)', y='Geometry Metric', 
             estimator='mean', errorbar='sd', 
            **lineplot_kwargs, 
             label=r'$nf/\|W\|_F^2$')
plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='y=0.5')
plt.legend()

plt.gca().set(ylabel='Geometry Metric $nf/\|W\|_F^2$ (mean ± sd)', 
              xlabel='1 / (Feature Probability) ', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "geometry_stability_seaborn", show=True)

# plot now all available instances

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='1/(1-S)', y='Geometry Metric', 
             hue='Instance Index', style='Instance Index',
             estimator='mean', errorbar='sd', 
             **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Geometry Metric $nf/∥W∥^2_F$ (mean ± sd)', 
              xlabel='1 / (Feature Probability) ', xscale='log')
logger.save_figure(plt.gcf(), "geometry_stability_seaborn_all_instances", show=True)

# %%
import masterthesis.plotting_utils as plt_utils

plt_utils.disp_color_df(df)


# %%
