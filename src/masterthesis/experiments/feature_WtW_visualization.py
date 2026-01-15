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
from masterthesis.toy_models_metrics import (
    count_represented_features_kwargs,
    count_represented_features_via_weight_kwargs
)
from masterthesis.icfg import ICfg, Expr
from masterthesis.trainer import Trainer

"""
Reproduce the feature visualization plot from toy models.
We plot W^T W as heatmap, where W is the weight matrix of a stacked coupled linear model. 
We also plot the weight norms in a separate vertical bar plot, where each bar is colored according to feature interference.
"""

""" 
n_features = 100,
n_hidden = 20,
n_instances = 20,

model = Model(
    # Exponential feature importance curve from 1 to 1/100
    importance = (100 ** -torch.linspace(0, 1, config.n_features))[None, :],
    # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None] 
"""

def single_run_cfg(base_cfg):
    
    # Set up sparsity curve matching original experiment
    # repeat each sparsity 8 times to calculate standard deviation.
    n_sparsities = 20
    n_repeats = 8
    original_sparsity = (20 ** -th.linspace(0, 1, n_sparsities))[:, None]
    repeated_feature_probability = original_sparsity.squeeze(1).repeat_interleave(n_repeats)[:, None]
    
    return ICfg(
        model=ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=n_sparsities * n_repeats,
            n_features=100,
            n_hidden=20,
            activation=nn.ReLU,
            bias=True
        ),
        trainer=ICfg(
            cls=Trainer,
            n_iterations=10_000 if not base_cfg.test else 100,
            optimizer_cls=optim.AdamW,
            optimizer_kwargs=ICfg(
                lr=1e-3,
                weight_decay=1e-2
            ),
        ),
        data=ICfg(
            n_batch=1024,
            feature_probability=repeated_feature_probability,  # Use repeated sparsity curve
            #importances=Expr('(0.9**th.arange(5))[None, :]'),
            importances=Expr('(100 ** -th.linspace(0, 1, 100))[None, :]'),
            feature_generator_type='independent_sparse',
            epsilon=0.1,
            min_norm=0.4,
            # Store original sparsity curve for plotting
            original_sparsity=original_sparsity.squeeze(1),
            n_sparsities=n_sparsities,
            n_repeats=n_repeats
        )
    )

def afun_base_cfg(afuns, name='feature_representation_over_afun', test = False):
    """Define the toplevel configuration for the whole experiment."""
    cfg = ICfg(
        name=name,
        tmp=False,
        test=test,
        run_cfg=single_run_cfg,
        run_args=afuns
    )
    if test:
        cfg.tmp = True

    return cfg

def run(run_cfg, device):
    cfg = run_cfg.eval()
    model = cfg.model.to(device)
    trainer = cfg.trainer
    data_iterator = instantiate_feature_generator(cfg.data.feature_generator_type, use_kwargs=True)
    feature_gen_kwargs = {
        'n_batch': cfg.data.n_batch,
        'n_features': model.n_features,
        'n_instances': model.n_instances,
        'feature_probability': cfg.data.feature_probability,
        'device': device
    }
    feature_generator = data_iterator(**feature_gen_kwargs)
    trainer.train(model, feature_generator, cfg.data.importances)

    return dict(
        model=model,
        trainer=trainer
    )

""" def render_features(model, which=np.s_[:]):
  cfg = model.config
  W = model.W.detach()
  W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

  interference = torch.einsum('ifh,igh->ifg', W_norm, W)
  interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

  polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
  net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
  norms = torch.linalg.norm(W, 2, dim=-1).cpu()

  WtW = torch.einsum('sih,soh->sio', W, W).cpu()

  # width = weights[0].cpu()
  # x = torch.cumsum(width+0.1, 0) - width[0]
  x = torch.arange(cfg.n_features)
  width = 0.9

  which_instances = np.arange(cfg.n_instances)[which]
"""

def calc_interference(model, cfg):
    W = model.layers['L_down'].W.detach()
    W_norm = W / (1e-5 + th.linalg.norm(W, 2, dim=-1, keepdim=True))
    interference = th.einsum('ifh,igh->ifg', W_norm, W)
    interference[:, th.arange(cfg.n_features), th.arange(cfg.n_features)] = 0
    polysemanticity = th.linalg.norm(interference, dim=-1).cpu()
    net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
    norms = th.linalg.norm(W, 2, dim=-1).cpu()
    WtW = th.einsum('sih,soh->sio', W, W).cpu()
    return WtW, polysemanticity, net_interference, norms

# calculate averages over the respective repeats

def calc_averages(model, cfg):
    WtW, polysemanticity, net_interference, norms = calc_interference(model, cfg)

    # WtW # shape (n_instances = n_sparsities * n_repeats, n_features, n_features)
    WtW_rsh = WtW.reshape(cfg.data.n_sparsities, cfg.data.n_repeats, cfg.data.n_features, cfg.data.n_features).mean(1)
    polysemanticity_rsh = polysemanticity.reshape(cfg.data.n_sparsities, cfg.data.n_repeats, cfg.data.n_features).mean(1)
    net_interference_rsh = net_interference.reshape(cfg.data.n_sparsities, cfg.data.n_repeats, cfg.data.n_features).mean(1)
    norms_rsh = norms.reshape(cfg.data.n_sparsities, cfg.data.n_repeats).mean(1)

    WtW_avg, WtW_std = WtW_rsh.mean(dim = 1), WtW_rsh.std(dim = 1)
    polysemanticity_avg, polysemanticity_std = polysemanticity_rsh.mean(dim = 1), polysemanticity_rsh.std(dim = 1)
    net_interference_avg, net_interference_std = net_interference_rsh.mean(dim = 1), net_interference_rsh.std(dim = 1)
    norms_avg, norms_std = norms_rsh.mean(dim = 1), norms_rsh.std(dim = 1)

    return dict(
        WtW=(WtW_avg, WtW_std),
        polysemanticity=(polysemanticity_avg, polysemanticity_std),
        net_interference=(net_interference_avg, net_interference_std),
        norms=(norms_avg, norms_std)
    )

def plot_results(logger:ExpLogger, model, cfg, device):
    results = calc_averages(model, cfg)

    WtW_avg, WtW_std = results['WtW']
    polysemanticity_avg, polysemanticity_std = results['polysemanticity']
    net_interference_avg, net_interference_std = results['net_interference']
    norms_avg, norms_std = results['norms']

    # Plot mean and std of W^T W as seaborn heatmap
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    sns.heatmap(WtW_avg, annot=False, cmap='coolwarm', cbar=True, square=True,
                xticklabels=cfg.data.original_sparsity, yticklabels=cfg.data.original_sparsity)
    plt.title('Mean of W^T W')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(logger.get_path('WtW_mean.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(WtW_std, annot=False, cmap='coolwarm', cbar=True, square=True,
                xticklabels=cfg.data.original_sparsity, yticklabels=cfg.data.original_sparsity)
    plt.title('Std of W^T W')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    #! save as svg
    plt.close()

    # Plot polysemanticity as seaborn bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cfg.data.original_sparsity, y=polysemanticity_avg, yerr=polysemanticity_std, palette='viridis')
    plt.title('Polysemanticity of Features')
    plt.xlabel('Feature Sparsity')
    plt.ylabel('Polysemanticity')
    #! save as svg
    plt.close()

    
    
    




