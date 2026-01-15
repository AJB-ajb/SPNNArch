#%% Asymptotic Feature Representation
# investigate the maximum number of represented features as a function of the number of hidden dimensions.
# note that the 'training pressure' is an important factor here, so we cannot just increase the number of features to an arbitrary number, but have to keep it bounded depending on the number of hidden dimensions or use importance decay.
# note also that the importance decay cannot be too strong, otherwise the model will only represent the most important features
# we measure features represented by their activation patterns at the end of training
# We choose the number of hidden dimensions to be 5, 10, 15, 20, …, 100
# Hypothesis: superlinear growth of represented features with respect to hidden dimensions, possibly quadratic or exponential
# we use LeakyReLU(1e-3) as activation function to avoid dead neurons and make the optimization problem more stable
# We use a different feature probabilities here, [1.0, 0.3, 0.1, 0.03, 0.01], with four instances per feature probability. For evaluation, we take the maximum number of represented features across all instances for each respective feature probability.

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
from masterthesis.toy_models_metrics import *
import masterthesis.toy_models_metrics as metrics
from masterthesis.experiment_logger import ExpLogger
from masterthesis import utils

# ICfg experiment config
def get_cfg(n_hidden, n_features):
    n_repeats = 4
    feature_probabilities = [0.1, 0.03, 0.01]
    n_instances = len(feature_probabilities) * n_repeats
    repeated_feature_probability = np.repeat(feature_probabilities, n_repeats)[:, None]
    return ICfg(
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
            n_iterations=10_000,
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
        feat_weight_metric=ICfg(cls=metrics.RepresentedFeatureWeightMetric, min_norm=0.4),
        feat_metric=ICfg(cls=metrics.RepresentedFeatureMetric, epsilon=0.1),
        #! unimplemented yet; compare feature representations for different error tolerances
        #! possibly have to update ICfg to allow eval for lists
        feat_metrics = [
            ICfg(cls=metrics.RepresentedFeatureMetric, epsilon=0.1),
            ICfg(cls=metrics.RepresentedFeatureMetric, epsilon=0.2),
            ICfg(cls=metrics.RepresentedFeatureMetric, epsilon=0.3),
        ],
        n_features=n_features,
        n_instances=n_instances,
        repeated_feature_probability=repeated_feature_probability,
        feature_probabilities=feature_probabilities,
    )

# Main experiment
LOAD = False
exp_name = 'asymptotic_feat_repr_icfg_large'
logger = ExpLogger(exp_name, tmp=False)

hidden_dims = list(range(5, 41, 5))
nfeatures1 = [n_hidden * 16 for n_hidden in hidden_dims]

if not LOAD:
    utils.set_seed()
    models = []
    configs = []
    trainers = []
    for n_hidden, n_features in zip(hidden_dims, nfeatures1):
        cfg = get_cfg(n_hidden, n_features)
        ecfg = cfg.eval()
        trainer = ecfg.trainer
        model = ecfg.model.to(utils.get_free_gpu())
        trainer.train(model)
        models.append(model)
        configs.append(ecfg)
        trainers.append(trainer)
    logger.save_data({'models': models, 'configs': configs, 'trainers': trainers}, 'models.pkl')
else:
    logger = ExpLogger.load_logger(exp_name)
    data = logger.load_data('models.pkl')
    models = data['models']
    configs = data['configs']
    trainers = data['trainers']

#%% 

results = []
for model, ecfg, trainer in zip(models, configs, trainers):
    represented_weight = ecfg.feat_weight_metric(model)
    represented_features = ecfg.feat_metric(model, n_features=ecfg.model.n_features, n_instances=ecfg.model.n_instances)
    
    # Multiple epsilon metrics
    epsilon_results = {}
    for feat_metric in ecfg.feat_metrics:
        epsilon = feat_metric.epsilon
        repr_feat = feat_metric(model, n_features=ecfg.model.n_features, n_instances=ecfg.model.n_instances)
        epsilon_results[f'Represented Features ({epsilon})'] = repr_feat
    
    # Convergence metric: relative std dev over last 100 iterations
    conv_metric = np.std(trainer.loss_history[-100:]) / trainer.loss_history[0] if len(trainer.loss_history) >= 100 else np.inf

    instance_idx = np.arange(ecfg.model.n_instances).repeat(ecfg.repeated_feature_probability.shape[0])
    for idx, prob in enumerate(ecfg.repeated_feature_probability):
        result = {
            'instance_idx': instance_idx[idx],
            'Hidden Dimensions': ecfg.model.n_hidden,
            'Num Features': ecfg.model.n_features,
            'Feature Probability': round(float(prob), 3),
            'Represented Features Weight': int(represented_weight[idx]),
            'Represented Features': int(represented_features[idx]),
            'Convergence Metric': conv_metric,
        }
        # Add epsilon results
        for col_name, repr_feat in epsilon_results.items():
            result[col_name] = int(repr_feat[idx])
        results.append(result)

if isinstance(results, list):
    df = pd.DataFrame(results)
else:
    df = results

#%%
import masterthesis.plotting_utils as plt_utils
plt_utils.disp_color_df(df)


#%%

# Calculate, for each feature probability, the maximum number of represented features ever observed (across all n_features and hidden dims)

max_repr = df.groupby(['Feature Probability', 'Hidden Dimensions'])['Represented Features'].max().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=max_repr,
    x='Hidden Dimensions',
    y='Represented Features',
    hue='Feature Probability',
    marker='o',
    legend='full',
)
plt.gca().set(ylabel='Max Represented Features', xlabel='Hidden Dimensions', xlim=(0, None), ylim=(0, None))
plt.title('Max Represented Features vs Hidden Dimensions (Per Feature Probability)')
plt.tight_layout()
logger._save_figure(plt.gcf(), "asymptotic_feat_repr_max_per_prob"); plt.show()

#%%
# Comparison plot across all epsilon metrics
epsilon_cols = [col for col in df.columns if col.startswith('Represented Features (')]
df_melted = df.melt(id_vars=['Feature Probability', 'Hidden Dimensions'], 
                    value_vars=epsilon_cols,
                    var_name='Epsilon Metric', value_name='Repr Features')

max_repr_all = df_melted.groupby(['Feature Probability', 'Hidden Dimensions', 'Epsilon Metric'])['Repr Features'].max().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=max_repr_all,
    x='Hidden Dimensions',
    y='Repr Features',
    hue='Feature Probability',
    style='Epsilon Metric',
    marker='o',
    legend='full',
)
plt.gca().set(ylabel='Max Represented Features', xlabel='Hidden Dimensions', xlim=(0, None), ylim=(0, None))
plt.title('Feature Representation Comparison Across Epsilon Thresholds')
plt.tight_layout()
logger._save_figure(plt.gcf(), "asymptotic_feat_repr_epsilon_comparison")
plt.show()

# %%

