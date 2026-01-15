#%% Optimizer Comparison - Side-by-Side Small vs Large Experiments
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from masterthesis.experiment_logger import ExpLogger
from masterthesis.utils import to_np
from masterthesis.style_reference import lineplot_kwargs, errorbar_kwargs

base_name = "optimizer_comparison"

def load_and_process_data(prefix):
    """Load and process optimizer comparison data for given prefix."""
    logger = ExpLogger.load_logger(base_name, prefix=prefix)
    models_dict, cfgs_dict = logger.load_models_cfgs_new()
    cfgs = list(cfgs_dict.values())
    models = [models_dict[cfg.label] for cfg in cfgs]
    
    results = []
    for model, cfg in zip(models, cfgs):
        ecfg = cfg.eval()
        instance_idx = np.arange(ecfg.n_sparsities).repeat(ecfg.n_repeats)
        
        represented_features = ecfg.feat_metric(model)
        represented_features_weight = ecfg.feat_weight_metric(model)
        
        run_df = pd.DataFrame({
            'label': ecfg.label,
            'instance_idx': instance_idx,
            'feature_probability': ecfg.trainer.feature_generator.feature_probability.squeeze().cpu().numpy(),
            'represented_features': to_np(represented_features),
            'represented_features_weight': to_np(represented_features_weight)
        })
        results.append(run_df)
    
    df = pd.concat(results, ignore_index=True)
    df_renamed = df.rename(columns={
        'label': 'Optimizer',
        'feature_probability': 'Feature Probability',
        'represented_features': 'Represented Features',
        'represented_features_weight': 'Represented Features Weight'
    })
    
    return df_renamed, logger

def plot_optimizer_sidebyside(small_df, large_df, filter_condition, title, filename, logger_small):
    """Create side-by-side plots for small and large experiments."""
    # Filter data
    small_filtered = small_df[small_df['Optimizer'].str.contains(filter_condition)]
    large_filtered = large_df[large_df['Optimizer'].str.contains(filter_condition)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Small experiment plot
    sns.lineplot(
        data=small_filtered, x='Feature Probability', y='Represented Features', 
        style='Optimizer', hue='Optimizer', markers=True,
        estimator='mean', errorbar='sd', legend='full',
        ax=ax1, **errorbar_kwargs, **lineplot_kwargs
    )
    ax1.set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', 
           xscale='log', title=f'{title} - Small (5 features)')
    ax1.grid(True, alpha=0.3)
    
    # Large experiment plot
    sns.lineplot(
        data=large_filtered, x='Feature Probability', y='Represented Features',
        style='Optimizer', hue='Optimizer', markers=True,
        estimator='mean', errorbar='sd', legend='full',
        ax=ax2, **errorbar_kwargs, **lineplot_kwargs
    )
    ax2.set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability',
           xscale='log', title=f'{title} - Large (100 features)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger_small.save_figure(fig, f"{filename}_sidebyside", show=True)
    return fig

#%% Load data for both experiments
print("Loading small experiment data...")
small_df, logger_small = load_and_process_data("small")

print("Loading large experiment data...")
large_df, logger_large = load_and_process_data("large")

#%% Create side-by-side plots for different optimizer categories

# Weight decay comparison
plot_optimizer_sidebyside(
    small_df, large_df, "wd", 
    "Weight Decay Comparison", "weight_decay_comparison", logger_small
)

# Learning rate comparison
plot_optimizer_sidebyside(
    small_df, large_df, "lr=", 
    "Learning Rate Comparison", "learning_rate_comparison", logger_small
)

# SGD momentum comparison
plot_optimizer_sidebyside(
    small_df, large_df, "momentum", 
    "SGD Momentum Comparison", "sgd_momentum_comparison", logger_small
)

# AdamW momentum (β₁) comparison
plot_optimizer_sidebyside(
    small_df, large_df, "β₁", 
    "AdamW Momentum Comparison", "adamw_momentum_comparison", logger_small
)

print("All side-by-side plots have been generated and saved!")
