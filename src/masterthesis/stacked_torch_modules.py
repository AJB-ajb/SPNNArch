import torch as th
import torch.nn as nn
import math
import tqdm
from deprecated import deprecated

class ShiftRescale(nn.Module):
    """
    Implements a simple affine scale and shift transformation with learnable parameters.
    The transformation is defined as:
        output = input * weight + bias
    where weight ∈ ℝ^n_instances and bias ∈ ℝ^n_instances (if bias=True)
    """
    def __init__(self, n_instances, bias=True):
        super().__init__()
        self.weight = nn.Parameter(th.Tensor(n_instances))
        self.bias = nn.Parameter(th.Tensor(n_instances)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: th.tensor):
        # input #(batch_size, n_instances, n_features)
        output = input * self.weight.unsqueeze(-1)
        if self.bias is not None:
            output += self.bias.unsqueeze(-1)
        return output

class StackedLinear(nn.Module):
    def __init__(self, n_instances, n_in, n_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(th.Tensor(n_instances, n_in, n_out))
        self.bias = nn.Parameter(th.Tensor(n_instances, n_out)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self): # note torch default; different than the one used in stacked linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, input: th.tensor):
        if input.dim() == 2: # if an input is given that doesn't vary among instances, add an instance dimension
            input = input.unsqueeze(1).expand(-1, self.weight.shape[0], -1)
        output = th.einsum('bid,ido->bio', input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

class StackedCoupledLinear(nn.Module):
    """
        Implements $
            x' = W^T W x + b
        $
    """
    def __init__(self, n_instances, n_in, n_hidden, bias=True):
        super().__init__()
        self.weight = nn.Parameter(th.empty(n_instances, n_in, n_hidden))
        self.bias = nn.Parameter(th.zeros(n_instances, n_in)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        # Bias is already initialized to zeros in __init__

    def forward(self, input: th.tensor):
        if input.dim() == 2:
            input = input.unsqueeze(1).expand(-1, self.weight.shape[0], -1)
        hidden = th.einsum('bid,idh->bih', input, self.weight)
        output = th.einsum('bih,idh->bid', hidden, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    
class StackedModel(nn.Module):
    """
        Abstract class for stacked models, i.e. models modeling n_instance smaller models at the same time for sake of efficiency in experiments
    """
    @deprecated(reason="Use WeightedMSELoss instead")
    def error(self, x):
        return th.mean((self(x) - x) ** 2, dim = (0, -1))
    @deprecated(reason="Use WeightedMSELoss instead")
    def loss(self, x):
        return th.mean(self.error(x))

    
class StackedLinearModel(StackedModel):
    """
    Implements $
        x' = α(W_up W_down x + b)
    $
    where α is the activation function
    """
    def __init__(self, n_instances, n_features, n_hidden, activation, bias = True, init_coupled = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activation = activation
        self.layers = nn.ModuleDict(
            dict(
                L_down = StackedLinear(n_instances, n_in=n_features, n_out=n_hidden, bias=False),
                L_up = StackedLinear(n_instances, n_in=n_hidden, n_out=n_features, bias=bias)
            )
        )
        
        # Initialize symmetrically if requested
        if init_coupled:
            self._init_coupled_weights()
        else:
            # Initialize with different strategies for encoder vs decoder
            with th.no_grad():
                # W_down: no activation after bottleneck → Xavier initialization
                nn.init.xavier_uniform_(self.layers['L_down'].weight)
                # W_up: feeds into activation → use PyTorch's built-in gain calculation
                functional_name = utils.get_functional_name(activation)
                gain = nn.init.calculate_gain(functional_name)
                nn.init.xavier_uniform_(self.layers['L_up'].weight, gain=gain)
    
    
    def _init_coupled_weights(self):
        """
        Initialize L_up weights as the transpose of L_down weights.
        This creates a symmetric initialization where W_up = W_down^T.
        """
        with th.no_grad():
            # L_down.weight shape: (n_instances, n_features, n_hidden)
            # L_up.weight shape: (n_instances, n_hidden, n_features)
            # Set L_up.weight = L_down.weight.transpose(-2, -1)
            self.layers['L_up'].weight.data = self.layers['L_down'].weight.data.transpose(-2, -1).clone()

    def forward(self, x):
        # x : [..., n_features]
        h = self.layers['L_down'](x)
        return self.activation(self.layers['L_up'](h))
    
    
class StackedCoupledLinearModel(StackedModel):
    """
    Implements $
        x' = α(W^T W x + b)
    $
    where α is the activation function
    """
    def __init__(self, n_instances, n_features, n_hidden, activation, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activation = activation
        self.layers = nn.ModuleDict(
            dict(
                L_down = StackedCoupledLinear(n_instances, n_in=n_features, n_hidden=n_hidden, bias=bias),
            )
        )

    def forward(self, x):
        return self.activation(self.layers['L_down'](x))


class StackedGatingModel(StackedModel):
    """
    Implements $
        x' = σ(W_σ h + b_σ) * act(W_up h + b)
    $
    where $h = W_down x$ via StackedLinear layers; 
    If coupled, then we have W_up = W_down^T.
    The activation function is applied to the output of the W_up layer and defaults to identity (none).
    If bias is True, then we add a bias term to the output of the W_up layer (This argument is roughly such that it behaves analogously to the one in StackedLinearModel, StackedCoupledLinearModel). 
    """
    def __init__(self, n_instances, n_features, n_hidden, gating_activation, activation = None, bias = True, coupled = False, init_σ_open = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.gating_activation = gating_activation
        self.activation = activation if activation is not None else nn.Identity()
        self.coupled = coupled
        
        if coupled:
            self.layers = nn.ModuleDict(
                dict(
                    L_down = StackedCoupledLinear(n_instances, n_in=n_features, n_hidden=n_hidden, bias=bias),
                    L_σ = StackedLinear(n_instances, n_in=n_hidden, n_out=n_features, bias=True)
                )
            )
        else:
            self.layers = nn.ModuleDict(
                dict(
                    L_down = StackedLinear(n_instances, n_in=n_features, n_out=n_hidden, bias=False),
                    L_up = StackedLinear(n_instances, n_in=n_hidden, n_out=n_features, bias=bias),
                    L_σ = StackedLinear(n_instances, n_in=n_hidden, n_out=n_features, bias=True)
                )
            )
        if init_σ_open:  # if requested, then we make sure that the gates are open by initializing the bias of L_σ to +3
            with th.no_grad():
                self.layers['L_σ'].bias.data.fill_(3.0)
        
    
    def forward(self, x):
        """
        forward pass
        """
        # x : [..., n_features]
        if self.coupled:
            # For coupled version: use W^T W x directly for output, compute h = W x for gating
            out = self.layers['L_down'](x)
            h = th.einsum('bid,idh->bih', x, self.layers['L_down'].weight)
        else:
            # For uncoupled version: standard two-layer approach
            h = self.layers['L_down'](x)
            out = self.layers['L_up'](h)
        
        σ = self.gating_activation(self.layers['L_σ'](h))
        out = σ * self.activation(out)
        return out
    
class WeightedMSELoss(nn.Module):
    """
    Computes the weighted mse loss for an instanced model.
    """
    def __init__(self, importances):
        super().__init__()
        self.register_buffer("importances", importances)

    def forward(self, x, y):
        """
        Forward pass for the loss computation.
        """
        loss = ((x - y) ** 2 * self.importances).mean(dim=(0, 2)).sum()
        return loss
    
    def instance_losses(self, x, y):
        """
        Compute the individual losses for each instance.
        """
        return ((x - y) ** 2 * self.importances).mean(dim=(0, 2))

# Optimization function for stacked modules
import masterthesis.utils as utils

def expand_instances(batch : th.tensor, n_instances):
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)
    if batch.dim() == 3:
        if batch.shape[1] == 1:
            return batch.expand(-1, n_instances, -1)
            
        assert batch.shape[1] == n_instances
        return batch
    
    raise NotImplementedError()



from deprecated import deprecated
@deprecated(reason="Use trainer class instead") 
def optimize(model, cfg):
    """
    Optimize the model and return the loss history
    
    Args:
        model: The model to optimize
        cfg: Configuration containing training parameters
        
    Returns:
        List of loss values during training
    """

    model = model.to(cfg['device'])
    optimizer_class = cfg.get('optimizer', th.optim.AdamW)
    optimizer = optimizer_class(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, **cfg.get('optimizer_kwargs', {}))
    loss_history = []
    weight_norms = []
    gradient_norms = []

    if cfg.get('lr_scheduler', None) is not None:
        scheduler = cfg.lr_scheduler(optimizer)
    else:
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0) 

    batch_iterator = cfg.feature_generator(cfg)
    
    for i in tqdm.tqdm(range(cfg.n_iterations)):
        x = next(batch_iterator)
        x = expand_instances(x, model.n_instances)  # Use model.n_instances instead of cfg.n_instances
        optimizer.zero_grad()
        y = model(x)
        # Reference loss: mean over batch and features, sum over instances
        loss = ((y - x)**2 * cfg.importances).mean(dim=(0,2)).sum()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        scheduler.step()

        if i % 100 == 0:
            weight_norms.append(th.norm(model.layers['L_down'].weight).item())
            gradient_norms.append(th.norm(th.cat([p.grad.flatten() for p in model.parameters()])).item())

        
    model.loss_history = loss_history
    model.weight_norms = weight_norms
    model.gradient_norms = gradient_norms

    # select model with minimal losses for a representative batch
    large_batch = next(cfg.feature_generator(cfg.replace(n_batch = 2048)))
    large_batch = expand_instances(large_batch, model.n_instances)  # Use model.n_instances

    model.losses = model.error(large_batch)
    model.i_min_loss = th.argmin(model.losses)
    model.min_loss = model.losses[model.i_min_loss]

    return loss_history