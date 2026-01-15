#%% [markdown]
"""
# Activation Function Comparison Experiment
This experiment compares various activation functions in terms of their impact on feature representation in a neural network model. The focus is on how different activation functions affect the number of represented features and their weights across varying sparsity levels.
"""

#%% Activation Function Comparison Experiment Using New Framework
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
from masterthesis.feature_generation import generate_ones_regular, IndependentSparseGen
from masterthesis.toy_models_metrics import RepresentedFeatureMetric, RepresentedFeatureWeightMetric
from masterthesis.style_reference import lineplot_kwargs, errorbar_kwargs


# Configuration for a single experimental run

# Set up sparsity curve matching original experiment
test = False
LOAD = True
prefix = "large"
base_name = "activation_function_comparison"

if prefix == "small":
    n_features = 5
    n_hidden = 2
    importances = Expr(f'(0.9**th.arange({n_features}))[None, :]')
    n_repeats = 16
elif prefix == "large":
    n_features = 100
    n_hidden = 10
    importances = Expr(f'(0.999**th.arange({n_features}))[None, :]')  # More gradual importance decay
    n_repeats = 4 # For the large models, the behavior is much more stable, so we can use fewer repeats
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
        activation=None,
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


# comparison of diverse activation functions
diverse_afuns = [
    (nn.ReLU(), "ReLU"),
    (nn.GELU(), "GELU"),
    (nn.SiLU(), "SiLU"),
    (nn.Tanh(), "Tanh"),
    (nn.Sigmoid(), "Sigmoid"),
]
# compare different leaky relu variants
leaky_relu_variants = [
    (nn.LeakyReLU(negative_slope=0.01), "LeakyReLU (0.01)"),
    (nn.LeakyReLU(negative_slope=0.1), "LeakyReLU (0.1)"),
    (nn.LeakyReLU(negative_slope=0.2), "LeakyReLU (0.5)"),
    (nn.LeakyReLU(negative_slope=0.5), "LeakyReLU (0.8)"),
    (nn.LeakyReLU(negative_slope=1.0), "LeakyReLU (1.0)"),
    (nn.LeakyReLU(negative_slope=1.5), "LeakyReLU (1.5)"),
]

# compare different ELU variants
elu_variants = [
    (nn.ELU(alpha=0.03), "ELU (0.03)"),
    (nn.ELU(alpha=0.1), "ELU (0.1)"),
    (nn.ELU(alpha=0.3), "ELU (0.3)"),
    (nn.ELU(alpha=1.0), "ELU (1.0)"),
    (nn.ELU(alpha=3.0), "ELU (3.0)"),
]

all_activation_data = [
    *diverse_afuns,
    *leaky_relu_variants,
    *elu_variants
]
#%%

# train all activation functions
import tqdm

def train_all(logger : ExpLogger, all_activation_data):
    models = []
    cfgs = [] 
    utils.set_seed()  
    for activation_data in all_activation_data: #tqdm.tqdm(all_activation_data, desc="Training models with different activation functions"):
        activation_fn, label = activation_data
        
        run_cfg = template_cfg.replace(label=label)
        run_cfg.model.activation = activation_fn
        
        eval_cfg = run_cfg.eval()
        
        model = eval_cfg.model.to(utils.get_free_gpu())
        print(f"Training model with activation: {run_cfg.label}")
        
        trainer = eval_cfg.trainer
        
        trainer.train(model)
        
        models.append(model)
        cfgs.append(run_cfg)


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
    train_all(logger, all_activation_data) 
else:
    logger = ExpLogger.load_logger(base_name, prefix=prefix)


# load the results and compute metrics 

import pandas as pd
import numpy as np
from masterthesis.utils import to_np

# Load models and configs using new ICfg-based method
models_dict, cfgs_dict = logger.load_models_cfgs_new()
cfgs = list(cfgs_dict.values())

# Convert back to lists for compatibility with existing code
models = [models_dict[cfg.label] for cfg in cfgs]

results = []
batch = None
for model, cfg in zip(models, cfgs):
    ecfg = cfg.eval()
    instance_idx = np.arange(ecfg.n_sparsities).repeat(ecfg.n_repeats)  # [sparsities * repeats]
    if batch is None:
        batch = ecfg.trainer.feature_generator(n_batch = ecfg.trainer.n_batch, n_features=ecfg.model.n_features, n_instances=ecfg.model.n_instances, device = next(ecfg.model.parameters()).device)

    feat_metric = ecfg.feat_metric
    feat_weight_metric = ecfg.feat_weight_metric
    
    represented_features = feat_metric(model)
    represented_features_weight = feat_weight_metric(model)
    
    feature_probability = to_np(ecfg.trainer.feature_generator.feature_probability).flatten()

    loss = ecfg.trainer.loss.instance_losses(
        model.forward(batch),
        batch
    )

    run_df = pd.DataFrame({
        'label': ecfg.label,
        'instance idx': np.arange(ecfg.model.n_instances),
        'sparsity group': instance_idx,
        'feature probability': feature_probability,
        'represented features': to_np(represented_features),
        'represented features via weight': to_np(represented_features_weight),
        'loss': to_np(loss)
    })
    results.append(run_df)


df = pd.concat(results, ignore_index=True)


# Save the DataFrame to a CSV file
df.to_csv(logger.path("activation_function_comparison_results.csv"), index=False)

# construct a dataframe which contains mean and std of represented features and weights

stats_df = df.groupby(['label', 'feature probability']).agg({
    'represented features': ['mean', 'std'],
    'represented features via weight': ['mean', 'std']
}).reset_index()

from masterthesis.plotting_utils import *

disp_color_df(stats_df)

#%%

# Add seaborn plots for activation function comparison
import seaborn as sns
import matplotlib.pyplot as plt

# Create lookup dictionaries for filtering
diverse_afun_labels = ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"]
leaky_relu_labels = ["ReLU"] + [label for _, label in leaky_relu_variants]
elu_labels = ["ReLU"] + [label for _, label in elu_variants]

df_renamed = df.rename(columns={
    'label': 'Activation Function',
    'feature probability': 'Feature Probability',
    'represented features': 'Represented Features',
    'represented features via weight': 'Represented Features Weight'
})

# Plot for Leaky ReLU variants
leaky_df = df_renamed[df_renamed['Activation Function'].isin(leaky_relu_labels)]
plt.figure(figsize=(10, 6))
sns.lineplot(data=leaky_df, x='Feature Probability', y='Represented Features Weight', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, **errorbar_kwargs, **lineplot_kwargs)

plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "leaky_relu_variants_comparison_weight_based", show=True)

#%% Plot losses of the leaky ReLU variants
# Calculate losses relative to minimum loss for each feature probability
leaky_df_loss = leaky_df.copy()
min_losses = leaky_df_loss.groupby('Feature Probability')['loss'].min().reset_index()
min_losses.columns = ['Feature Probability', 'min_loss']
leaky_df_loss = leaky_df_loss.merge(min_losses, on='Feature Probability')
leaky_df_loss['Loss Diff'] = leaky_df_loss['loss'] - leaky_df_loss['min_loss']

# Take ReLU model with minimum loss and evaluate with different activation functions
relu_model = models_dict["ReLU"]
relu_cfg = cfgs_dict["ReLU"].eval()

activation_swap_results = []
for fp in leaky_df['Feature Probability'].unique():
    # Find best ReLU instance for this feature probability
    relu_instances = leaky_df[(leaky_df['Activation Function'] == 'ReLU') & 
                             (leaky_df['Feature Probability'] == fp)]
    best_idx = relu_instances['loss'].idxmin()
    instance_idx = relu_instances.loc[best_idx, 'instance idx']
    
    # Evaluate with different LeakyReLU activations
    for activation_fn, label in leaky_relu_variants:
        # Create temporary model with swapped activation
        temp_model = type(relu_model)(
            n_instances=relu_model.n_instances,
            n_features=relu_model.n_features, 
            n_hidden=relu_model.n_hidden,
            activation=activation_fn,
            bias=True
        )
        
        # Copy weights from best ReLU model
        temp_model.load_state_dict(relu_model.state_dict())
        
        # Evaluate loss for this instance
        with th.no_grad():
            output = temp_model(batch)
            loss = relu_cfg.trainer.loss.instance_losses(output, batch)[instance_idx]
        
        activation_swap_results.append({
            'Feature Probability': fp,
            'Activation Function': label,
            'loss': loss.item()
        })

swap_df = pd.DataFrame(activation_swap_results)

# Side-by-side loss comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Loss difference from minimum
sns.lineplot(data=leaky_df_loss, x='Feature Probability', y='Loss Diff', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, 
             ax=ax1, **errorbar_kwargs, **lineplot_kwargs)
ax1.set(ylabel='Loss - Min Loss', xlabel='Feature Probability', xscale='log', yscale='log', title='Loss - Min Loss Over Models')

# Right plot: ReLU weights with different activations
sns.lineplot(data=swap_df, x='Feature Probability', y='loss', 
             hue='Activation Function', style='Activation Function',
             markers=True, legend=False, ax=ax2, **lineplot_kwargs)
ax2.set(ylabel='Loss (ReLU weights)', xlabel='Feature Probability', xscale='log', yscale='log', title='Loss of Different Activations on Best ReLU Weights')

plt.tight_layout()
logger.save_figure(fig, "leaky_relu_loss_comparison_sidebyside", show=True)


#%%

# Plot for ELU variants
elu_df = df_renamed[df_renamed['Activation Function'].isin(elu_labels)]
plt.figure(figsize=(10, 6))
sns.lineplot(data=elu_df, x='Feature Probability', y='Represented Features Weight', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "elu_variants_comparison_weight_based", show=True)
#%%

# Plot for diverse activation functions
diverse_df = df_renamed[df_renamed['Activation Function'].isin(diverse_afun_labels)]
plt.figure(figsize=(10, 6))
sns.lineplot(data=diverse_df, x='Feature Probability', y='Represented Features Weight', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "diverse_activation_functions_comparison_weight_based", show=True)

# Plot for ReLU, SiLU, and GELU
relu_silu_gelu_df = df_renamed[df_renamed['Activation Function'].isin(["ReLU", "SiLU", "GELU"])]
plt.figure(figsize=(10, 6))
sns.lineplot(data=relu_silu_gelu_df, x='Feature Probability', y='Represented Features Weight', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
relu_variants_fig = plt.gcf()
logger.save_figure(plt.gcf(), "relu_variants_comparison", show=True)

# %%
# Side-by-side combination of activation functions with ReLU variants

def plot_activation_comparison_sidebyside(df_renamed, activation_labels, title_prefix, filename_prefix, logger, x_range=(-3, 3), num_points=1000):
    """
    Create side-by-side plots showing feature representation and activation function shapes.
    
    Args:
        df_renamed: DataFrame with renamed columns for plotting
        activation_labels: List of activation function labels to include
        title_prefix: Prefix for the plot titles
        filename_prefix: Prefix for the saved filename
        logger: ExpLogger instance for saving figures
        x_range: Range of x values for activation function plotting
        num_points: Number of points to use for activation function curves
    """
    # Filter data for the specified activation functions
    filtered_df = df_renamed[df_renamed['Activation Function'].isin(activation_labels)]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Feature representation results
    sns.lineplot(data=filtered_df, x='Feature Probability', y='Represented Features Weight', 
                hue='Activation Function', style='Activation Function', 
                estimator='mean', errorbar='sd', legend='full', markers=True, 
                ax=ax1, **errorbar_kwargs, **lineplot_kwargs)
    
    ax1.set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')

    ax1.set_title(f'{title_prefix} - Feature Representation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Activation function shapes using seaborn
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Create DataFrame for activation function shapes
    activation_data = []
    for label in activation_labels:
        # Create the activation function based on the label
        if label == "ReLU":
            y = np.maximum(0, x)
        elif label == "GELU":
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            y = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        elif label == "SiLU":
            # SiLU: x * sigmoid(x)
            y = x / (1 + np.exp(-x))
        elif label == "Tanh":
            y = np.tanh(x)
        elif label == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
        elif "LeakyReLU" in label:
            # Extract slope from label
            slope = float(label.split('(')[1].split(')')[0])
            y = np.where(x > 0, x, slope * x)
        elif "ELU" in label:
            # Extract alpha from label
            alpha = float(label.split('(')[1].split(')')[0])
            y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        else:
            continue  # Skip unknown activation functions
        
        # Add to activation data
        for i, (x_val, y_val) in enumerate(zip(x, y)):
            activation_data.append({
                'Input': x_val,
                'Output': y_val,
                'Activation Function': label
            })
    
    # Create DataFrame and plot with seaborn
    activation_df = pd.DataFrame(activation_data)
    sns.lineplot(data=activation_df, x='Input', y='Output', hue='Activation Function', style='Activation Function',
                ax=ax2, **lineplot_kwargs, legend=None)

    ax2.set(xlabel='Input', ylabel='Output', title = f'{title_prefix} - Activation Functions')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    logger.save_figure(fig, f"{filename_prefix}_sidebyside_comparison", show=True)
    return fig

# Create side-by-side plots for different activation function groups

# 1. Diverse activation functions
plot_activation_comparison_sidebyside(
    df_renamed, diverse_afun_labels, 
    "Diverse Activation Functions", "diverse_activation_functions", logger
)
# note: technically, we should change the initialization according to the activation function and learn scaled reparameterization

# 2. Leaky ReLU variants
plot_activation_comparison_sidebyside(
    df_renamed, leaky_relu_labels, 
    "Leaky ReLU Variants", "leaky_relu_variants", logger
)

# 3. ELU variants
plot_activation_comparison_sidebyside(
    df_renamed, elu_labels, 
    "ELU Variants", "elu_variants", logger
)

# 4. ReLU, SiLU, and GELU comparison
plot_activation_comparison_sidebyside(
    df_renamed, ["ReLU", "SiLU", "GELU"], 
    "ReLU vs Modern Activations", "relu_modern_comparison", logger
)