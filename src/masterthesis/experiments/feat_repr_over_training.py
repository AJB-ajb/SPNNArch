#%% Feature Representation Over Training Experiment

# Compare the feature representations over training
# how does the feature representation change over training?
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from masterthesis.icfg import ICfg, Expr
from masterthesis.trainer import Trainer
from masterthesis.stacked_torch_modules import StackedCoupledLinearModel, WeightedMSELoss
from masterthesis.feature_generation import IndependentSparseGen
from masterthesis.toy_models_metrics import RepresentedFeatureMetric, RepresentedFeatureWeightMetric
from masterthesis.experiment_logger import ExpLogger
from masterthesis import utils

# Hook factory

def make_hook(df, feat_metric, feat_weight_metric, feature_probability, hook_interval):
    # Generate a single batch that will be reused for all loss evaluations
    evaluation_batch = None
    
    def hook(model, step, trainer):
        nonlocal evaluation_batch
        
        if step % hook_interval == 0:
            rf = feat_metric(model)
            rfw = feat_weight_metric(model)
            
            # Generate the evaluation batch only once, on the first call
            if evaluation_batch is None:
                device = next(model.parameters()).device
                evaluation_batch = trainer.feature_generator(
                    n_batch=trainer.n_batch,
                    n_features=model.n_features,
                    n_instances=model.n_instances,
                    device=device
                )
            
            # Compute instance losses using the same batch every time
            with th.no_grad():
                model_output = model(evaluation_batch)
                instance_losses = trainer.loss.instance_losses(model_output, evaluation_batch)
            
            n_instances = len(rf)  # Get n_instances from the length of results
            for i in range(n_instances):
                df.append({
                    'Step': step,
                    'Instance': i,
                    'Feature Probability': round(float(feature_probability[i]), 3),
                    'Represented Features': int(rf[i]),
                    'Represented Features Weight': int(rfw[i]),
                    'Loss': float(instance_losses[i]),
                })
    return hook

# ICfg experiment config

LOAD = False
test = False
prefix = "small"  # or "large" for large experiment
base_name = "feat_repr_over_training_icfg"

if prefix == "small":
    n_features = 5
    n_hidden = 2
    n_repeats = 16
    importances = Expr(f'(0.9**th.arange({n_features}))[None, :]')
elif prefix == "large":
    n_features = 100
    n_hidden = 10
    n_repeats = 16
    importances = Expr(f'(0.999**th.arange({n_features}))[None, :]')  # More gradual importance decay
else:
    raise ValueError(f"Unknown prefix: {prefix}")

n_sparsities = 6
n_instances = n_sparsities * n_repeats
original_sparsity = (20 ** -th.linspace(0, 1, n_sparsities))[:, None]
repeated_feature_probability = original_sparsity.squeeze(1).repeat_interleave(n_repeats)[:, None]

template_cfg = ICfg(
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
        n_batch=1024,
        n_iterations=10_000 if not test else 10,
        log_interval=100,
        optimizer_cls=optim.AdamW,
        optimizer_kwargs=ICfg(lr=1e-3, weight_decay=1e-2),
        loss=ICfg(cls=WeightedMSELoss, importances=importances),
        feature_generator=ICfg(
        cls=IndependentSparseGen,
        feature_probability=repeated_feature_probability,
        distribution='uniform'
    )
    ),
    feat_metric=ICfg(cls=RepresentedFeatureMetric, epsilon=0.2, n_features=n_features, n_instances=n_instances),
    feat_weight_metric=ICfg(cls=RepresentedFeatureWeightMetric, min_norm=0.4),
    n_features=n_features,
    n_instances=n_instances,
    repeated_feature_probability=repeated_feature_probability,
    hook_interval=100,
)

# Main experiment

if not LOAD:
    logger = ExpLogger(
        base_name=base_name,
        tmp=test,
        prefix=prefix,
    )
    utils.set_seed()
    df = []
    cfg = template_cfg
    ecfg = cfg.eval()
    hook = make_hook(df, ecfg.feat_metric, ecfg.feat_weight_metric, ecfg.repeated_feature_probability, ecfg.hook_interval)
    trainer = ecfg.trainer
    trainer.hook = hook
    model = ecfg.model.to(utils.get_free_gpu())
    trainer.train(model)
    logger.save_model(model, 'model.pt')
    logger.save_data(df, 'df.pkl')
else:
    logger = ExpLogger.load_logger(base_name, prefix=prefix)
    cfg = template_cfg
    model = logger.load_model(cfg.eval().model.__class__, 'model.pt')
    df = logger.load_data('df.pkl')

if isinstance(df, list):
    df = pd.DataFrame(df)

#%%
from masterthesis.style_reference import lineplot_kwargs, errorbar_kwargs

# Filter early training data
df_early = df[df['Step'] <= 3000]

# Side-by-side plots for feature representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Full training plot
sns.lineplot(data=df, x='Step', y='Represented Features', hue='Feature Probability', style='Feature Probability', 
             estimator='mean', errorbar='sd', legend='full', ax=ax1, **errorbar_kwargs, **lineplot_kwargs)
ax1.set(ylabel='Represented Features (mean ± sd)', xlabel='Training Step', title='Full Training')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Early training plot  
sns.lineplot(data=df_early, x='Step', y='Represented Features', hue='Feature Probability', style='Feature Probability',
             estimator='mean', errorbar='sd', legend=None, ax=ax2, **errorbar_kwargs, **lineplot_kwargs)
ax2.set(ylabel='Represented Features (mean ± sd)', xlabel='Training Step', title='Early Training (≤3000 steps)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
logger.save_figure(fig, "feat_repr_over_training_sidebyside")
plt.show()

# Side-by-side plots for weight-based feature representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Full training plot
sns.lineplot(data=df, x='Step', y='Represented Features Weight', hue='Feature Probability', style='Feature Probability',
             estimator='mean', errorbar='sd', legend='full', ax=ax1, **errorbar_kwargs, **lineplot_kwargs)
ax1.set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Training Step', title='Full Training')
ax1.grid(True, alpha=0.3)

# Early training plot
sns.lineplot(data=df_early, x='Step', y='Represented Features Weight', hue='Feature Probability', style='Feature Probability',
             estimator='mean', errorbar='sd', legend=None, ax=ax2, **errorbar_kwargs, **lineplot_kwargs)
ax2.set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Training Step', title='Early Training (≤3000 steps)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
logger.save_figure(fig, "feat_repr_weight_over_training_sidebyside")
plt.show()

# Side-by-side plots for loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Full training plot
sns.lineplot(data=df, x='Step', y='Loss', hue='Feature Probability', style='Feature Probability',
             estimator='mean', errorbar='sd', legend='full', ax=ax1, **errorbar_kwargs, **lineplot_kwargs)
ax1.set(ylabel='Loss (mean ± sd)', xlabel='Training Step', yscale='log', title='Full Training')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Early training plot
sns.lineplot(data=df_early, x='Step', y='Loss', hue='Feature Probability', style='Feature Probability',
             estimator='mean', errorbar='sd', legend=None, ax=ax2, **errorbar_kwargs, **lineplot_kwargs)
ax2.set(ylabel='Loss (mean ± sd)', xlabel='Training Step', yscale='log', title='Early Training (≤3000 steps)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
logger.save_figure(fig, "loss_over_training_sidebyside")
plt.show()

#%% Individual plots (kept for compatibility)

plt.figure()
sns.lineplot(data=df, x='Step', y='Represented Features', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.legend(loc='upper left')
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "feat_repr_over_training")
plt.show()

plt.figure()
sns.lineplot(data=df_early, x='Step', y='Represented Features', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "feat_repr_over_training_early")
plt.show()

plt.figure()
sns.lineplot(data=df, x='Step', y='Represented Features Weight', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "feat_repr_weight_over_training")
plt.show()

plt.figure()
sns.lineplot(data=df_early, x='Step', y='Represented Features Weight', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "feat_repr_weight_over_training_early")
plt.show()

plt.figure()
sns.lineplot(data=df, x='Step', y='Loss', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Loss (mean ± sd)', xlabel='Training Step', yscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "loss_over_training")
plt.show()

plt.figure()
sns.lineplot(data=df_early, x='Step', y='Loss', hue='Feature Probability', style='Feature Probability', estimator='mean', errorbar='sd', legend='full', **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Loss (mean ± sd)', xlabel='Training Step', yscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "loss_over_training_early")
plt.show()

# %%
