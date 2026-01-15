#%% Scaling Over Training Experiment

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

def make_hook(df, feat_metric, feat_weight_metric, n_features, n_instances, feature_probability, hook_interval, model_size):
    """Hook to track feature representation during training"""
    def hook(model, step, trainer):
        if step % hook_interval == 0:
            rf = feat_metric(model, n_features, n_instances)
            rfw = feat_weight_metric(model)
            for i in range(n_instances):
                df.append({
                    'Step': step,
                    'Instance': i,
                    'Model Size': model_size,
                    'Feature Probability': round(float(feature_probability[i]), 3),
                    'Represented Features': int(rf[i]),
                    'Represented Features Weight': int(rfw[i]),
                })
    return hook

def single_size_cfg(n_features, n_hidden, model_size_label):
    """Create configuration for a single model size"""
    n_sparsities, n_repeats = 3, 16  # Reduced for faster experimentation
    n_instances = n_sparsities * n_repeats
    
    # Three sparsity levels: high, medium, low
    sparsity_levels = th.tensor([0.05, 0.2, 0.8])[:, None]
    repeated_feature_probability = sparsity_levels.squeeze(1).repeat_interleave(n_repeats)[:, None]
    
    return ICfg(
        label=model_size_label,
        n_features=n_features,
        n_instances=n_instances,
        repeated_feature_probability=repeated_feature_probability,
        
        model=ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=n_instances,
            n_features=n_features,
            n_hidden=n_hidden,
            activation=nn.LeakyReLU(1e-3),
            bias=True
        ),
        
        trainer=ICfg(
            cls=Trainer,
            n_batch=1024,
            n_iterations=5000,  # Reduced for faster training
            log_interval=100,
            optimizer_cls=optim.AdamW,
            optimizer_kwargs=ICfg(lr=1e-3, weight_decay=1e-2),
            loss=ICfg(cls=WeightedMSELoss, importances=Expr(f'(0.999**th.arange({n_features}))[None, :]')),
            feature_generator=ICfg(
                cls=IndependentSparseGen,
                feature_probability=repeated_feature_probability,
                distribution='uniform'
            )
        ),
        
        feat_metric=ICfg(cls=RepresentedFeatureMetric, epsilon=0.1),
        feat_weight_metric=ICfg(cls=RepresentedFeatureWeightMetric, min_norm=0.4),
        hook_interval=100,
    )

# Model sizes to compare
model_sizes = [
    (3, "3 hidden neurons"),
    (6, "6 hidden neurons"), 
    (9, "9 hidden neurons"),
    (12, "12 hidden neurons")
]

def train_all_sizes(logger, model_sizes):
    """Train models for all sizes and collect data"""
    df = []
    models = {}
    
    n_features = 30 # fixed number of features for all models
    for n_hidden, size_label in model_sizes:
        print(f"Training model with {size_label}...")
        
        utils.set_seed()  # Consistent initialization
        cfg = single_size_cfg(n_features, n_hidden, size_label)
        ecfg = cfg.eval()
        
        # Create hook for this model size
        hook = make_hook(df, ecfg.feat_metric, ecfg.feat_weight_metric, 
                        ecfg.n_features, ecfg.n_instances, 
                        ecfg.repeated_feature_probability, ecfg.hook_interval, size_label)
        
        # Setup and train
        model = ecfg.model.to(utils.get_free_gpu())
        trainer = ecfg.trainer
        trainer.hook = hook
        trainer.train(model)
        
        models[size_label] = model
    
    return pd.DataFrame(df), models

# Main experiment
LOAD = False
exp_name = 'scaling_hidden_over_training'
logger = ExpLogger(exp_name, tmp=False)

if not LOAD:
    df, models = train_all_sizes(logger, model_sizes)
    logger.save_models(models, 'models.pt')
    logger.save_data(df, 'df.pkl')
else:
    logger = ExpLogger.load_logger(exp_name)
    df = logger.load_data('df.pkl')

#%% Plotting Results

# Feature representation over training by hidden size
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Represented Features', hue='Model Size', 
             style='Feature Probability', estimator='mean', errorbar='sd',
             err_style='band', err_kws={'alpha': 0.1})
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "scaling_hidden_feat_repr_over_training")
plt.show()

# Weight-based representation over training by hidden size
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Represented Features Weight', hue='Model Size',
             style='Feature Probability', estimator='mean', errorbar='sd',
             err_style='band', err_kws={'alpha': 0.1})
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Training Step')
plt.tight_layout()
logger.save_figure(plt.gcf(), "scaling_hidden_feat_repr_weight_over_training")
plt.show()

# Final representation comparison by hidden size (last step only)
df_final = df[df['Step'] == df['Step'].max()]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_final, x='Model Size', y='Represented Features', hue='Feature Probability')
plt.gca().set(ylabel='Represented Features', xlabel='Hidden Layer Size')
plt.tight_layout()
logger.save_figure(plt.gcf(), "scaling_hidden_final_feat_repr")
plt.show()

print("Hidden layer scaling experiment completed!")

# %%
