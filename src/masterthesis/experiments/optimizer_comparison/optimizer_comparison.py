#%% Optimizer Comparison Experiment Using New Framework
import IPython.display
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import ray

from masterthesis.utils import *
from masterthesis.plotting_utils import *
from masterthesis.stacked_torch_modules import *
from masterthesis.experiment_logger import ExpLogger
from masterthesis.icfg import ICfg, Expr
from masterthesis.trainer import Trainer
from masterthesis.feature_generation import *
from masterthesis.toy_models_metrics import RepresentedFeatureMetric, RepresentedFeatureWeightMetric
from masterthesis.style_reference import lineplot_kwargs, errorbar_kwargs


# Configuration for a single experimental run


#optimizer_cls, optimizer_kwargs, label = optimizer_data

# Set up sparsity curve matching original experiment
test = False
LOAD = True
prefix = "small" # or "small" for small experiment
base_name = "optimizer_comparison"

if prefix == "small":
    n_features = 5
    n_hidden = 2
    n_repeats = 16
    importances = Expr(f'(0.9**th.arange({n_features}))[None, :]')
elif prefix == "large":
    n_features = 100
    n_hidden = 10
    importances = Expr(f'(0.999**th.arange({n_features}))[None, :]')  # More gradual importance decay
    n_repeats = 4
else:
    raise ValueError(f"Unknown prefix: {prefix}")

n_sparsities = 8
original_sparsity = (20 ** -th.linspace(0, 1, n_sparsities))[:, None]
repeated_feature_probability = original_sparsity.squeeze(1).repeat_interleave(n_repeats)[:, None]

n_instances = n_sparsities * n_repeats

template_cfg = ICfg(
    label=None,
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
        optimizer_cls=None,
        optimizer_kwargs=None,
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
    feat_metric = ICfg(
        cls=RepresentedFeatureMetric,
        epsilon=0.2,  # Threshold for feature representation
        n_features=n_features,
        n_instances=n_instances
    ),
    feat_weight_metric = ICfg(
        cls=RepresentedFeatureWeightMetric,
        min_norm=0.4  # Minimum norm for feature representation via weight
    ),
)

# Define the optimizers to compare
optimizers_wd = [
    # dependency on weight decay
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':0.0}, 'AdamW (wd=0.0)'),
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':1e-2}, 'AdamW (wd=1e-2)'),
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':1e-1}, 'AdamW (wd=1e-1)'),
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':1.0}, 'AdamW (wd=1.0)'),
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':10}, 'AdamW (wd=10.0)'),
]

optimizers_lr = [
    # dependency on learning rate
    (optim.AdamW, {'lr': 1e-4, 'weight_decay':1e-2}, 'AdamW (lr=1e-4)'),
    (optim.AdamW, {'lr': 1e-3, 'weight_decay':1e-2}, 'AdamW (lr=1e-3)'),
    (optim.AdamW, {'lr': 1e-2, 'weight_decay':1e-2}, 'AdamW (lr=1e-2)'),
    (optim.AdamW, {'lr': 1e-1, 'weight_decay':1e-2}, 'AdamW (lr=1e-1)'),
    (optim.AdamW, {'lr': 1.0, 'weight_decay':1e-2}, 'AdamW (lr=1.0)'),
]

# continuously remove momentum in AdamW
optimizers_adamw_momentum = [
    (optim.AdamW, {'lr': 1e-3, 'betas': (0. , 0.999)}, 'AdamW (β₁=0.0)'),
    (optim.AdamW, {'lr': 1e-3, 'betas': (1 - 3e-1, 0.999)}, 'AdamW (β₁=0.7)'),
    (optim.AdamW, {'lr': 1e-3, 'betas': (1 - 1e-1, 0.999)}, 'AdamW (β₁=0.9)'),
    (optim.AdamW, {'lr': 1e-3, 'betas': (1 - 3e-2, 0.999)}, 'AdamW (β₁=0.97)'),
    (optim.AdamW, {'lr': 1e-3, 'betas': (1 - 1e-2, 0.999)}, 'AdamW (β₁=0.99)'),
    (optim.AdamW, {'lr': 1e-3, 'betas': (1 - 3e-3, 0.999)}, 'AdamW (β₁=0.997)'),
]

# dependency on momentum for SGD
optimizers_momentum = [
    (optim.SGD, {'lr': 1e-3, 'momentum': 0.}, 'SGD (momentum=0.0)'),
    (optim.SGD, {'lr': 1e-3, 'momentum': 1-3e-1}, 'SGD (momentum=0.7)'),
    (optim.SGD, {'lr': 1e-3, 'momentum': 1-1e-1}, 'SGD (momentum=0.9)'),
    (optim.SGD, {'lr': 1e-3, 'momentum': 1-3e-2}, 'SGD (momentum=0.97)'),
    (optim.SGD, {'lr': 1e-3, 'momentum': 1-1e-2}, 'SGD (momentum=0.99)'),
]

# merge all optimizer configurations into a single list
all_optimizer_data = [
    *optimizers_wd,
    *optimizers_lr,
    *optimizers_adamw_momentum,
    *optimizers_momentum
]
     
#%%
# train all optimizers
import tqdm

def train_all(logger : ExpLogger, all_optimizer_data):
    models = []
    cfgs = [] 
    utils.set_seed()
    for optimizer_data in all_optimizer_data: #tqdm.tqdm(all_optimizer_data, desc="Training models with different optimizers"):
        optimizer_cls, optimizer_kwargs,label = optimizer_data
        
        run_cfg = template_cfg.replace(label=label)  
        run_cfg.trainer.optimizer_cls = optimizer_cls
        run_cfg.trainer.optimizer_kwargs = ICfg(**optimizer_kwargs)
        
        
        eval_cfg = run_cfg.eval()
        
        model = eval_cfg.model.to(utils.get_free_gpu())
        print(f"Training model with optimizer: {run_cfg.label}")
        
        trainer = eval_cfg.trainer
        
        trainer.train(model)
        
        models.append(model)
        cfgs.append(run_cfg)

        # Save individual models using proper model saving

    # Save all models together using the models dictionary function
    model_dict = {cfg.label: model for model, cfg in zip(models, cfgs)}
    cfg_dict = {cfg.label: cfg for cfg in cfgs}
    logger.save_models_cfgs_new(model_dict, cfg_dict)
    

if not LOAD:
    # create logger
    logger = ExpLogger(
        base_name = base_name,
        tmp = test,
        prefix = prefix,
    )
    train_all(logger, all_optimizer_data) 
else:
    logger = ExpLogger.load_logger(base_name, prefix=prefix)

#%%
import pandas as pd
import numpy as np
from masterthesis.utils import to_np

# Load models and configs using new ICfg-based method
models_dict, cfgs_dict = logger.load_models_cfgs_new()
cfgs = list(cfgs_dict.values())

# Convert back to lists for compatibility with existing code
models = [models_dict[cfg.label] for cfg in cfgs]

df = pd.DataFrame(columns=['label', 'instance_idx', 'feature_probability', 'represented_features', 'represented_features_weight'])

for model, cfg in zip(models, cfgs):
    ecfg = cfg.eval()
    instance_idx = np.arange(ecfg.n_sparsities).repeat(ecfg.n_repeats)  # [sparsities * repeats]
    trainer = ecfg.trainer

    represented_features = ecfg.feat_metric(model)
    represented_features_weight = ecfg.feat_weight_metric(model)

    df = pd.concat([df, pd.DataFrame({
        'label': ecfg.label,
        'instance_idx': instance_idx,
        'feature_probability': trainer.feature_generator.feature_probability.squeeze().cpu().numpy(),
        'represented_features': to_np(represented_features),
        'represented_features_weight': to_np(represented_features_weight)
    })], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(logger.path("optimizer_comparison_results.csv"), index=False)

# construct a dataframe which contains mean and std of represented features and weights
stats_df = df.groupby(['label', 'feature_probability']).agg({
    'represented_features': ['mean', 'std'],
    'represented_features_weight': ['mean', 'std']
}).reset_index()

# Flatten multi-level column index for easy plotting
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]


from masterthesis.plotting_utils import *

disp_color_df(stats_df)


#%% seaborn plots for the results
import seaborn as sns
import matplotlib.pyplot as plt

df_renamed = df.rename(columns={
    'label': 'Optimizer',
    'feature_probability': 'Feature Probability',
    'represented_features': 'Represented Features',
    'represented_features_weight': 'Represented Features Weight'
})

# Filter for weight decay experiments
wd_df = df_renamed[df_renamed['Optimizer'].str.contains("wd")]
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=wd_df, x='Feature Probability', y='Represented Features', style='Optimizer', 
    estimator='mean', errorbar='sd', legend='full', 
    hue='Optimizer', markers=True,
    **errorbar_kwargs, **lineplot_kwargs
)
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "optimizer_comparison_feat_repr", show = True)

# Filter for learning rate experiments
lr_df = df_renamed[df_renamed['Optimizer'].str.contains("lr=")]
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=lr_df, x='Feature Probability', y='Represented Features', style='Optimizer', hue='Optimizer', markers=True, 
    estimator='mean', errorbar='sd', legend='full', 
    **errorbar_kwargs, **lineplot_kwargs
)
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "optimizer_comparison_feat_repr_weight", show = True)

# Filter for momentum experiments
momentum_df = df_renamed[df_renamed['Optimizer'].str.contains("momentum")]
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=momentum_df, x='Feature Probability', y='Represented Features', style='Optimizer', hue='Optimizer',
    estimator='mean', errorbar='sd', legend='full', markers = True, 
    **errorbar_kwargs, **lineplot_kwargs
)
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "optimizer_comparison_momentum_based", show=True)

# Filter for AdamW momentum experiments
adamw_momentum_df = df_renamed[df_renamed['Optimizer'].str.contains("β₁")]
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=adamw_momentum_df, x='Feature Probability', y='Represented Features', style='Optimizer', hue='Optimizer',
    estimator='mean', errorbar='sd', legend='full', markers = True, 
    **errorbar_kwargs, **lineplot_kwargs
)
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "optimizer_comparison_adamw_momentum", show=True)




