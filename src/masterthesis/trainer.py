"""
Trainer class for training stacked models without cfg dependencies.

This module provides a clean trainer interface that can be instantiated via ICfg
and handles model optimization with proper separation of concerns.
"""

import torch
import tqdm
from .stacked_torch_modules import expand_instances
from .icfg import ICfg


class Trainer:
    """
    A trainer class that handles model optimization without requiring a cfg object.
    Designed to be instantiable via ICfg configurations.
    """
    
    def __init__(self, 
                 feature_generator, 
                 n_batch,
                 loss,                 
                 optimizer_cls,
                 optimizer_kwargs=None,
                 lr_scheduler_cls=None,
                 lr_scheduler_kwargs=None,
                 n_iterations=10000,
                 log_interval=None,
                 save_interval=None,
                 hook=None,
                 epsilon_early_stopping=None,
                 n_patience=None):
        """
        Initialize trainer with all necessary parameters.
        
        Args:
            optimizer_cls: Optimizer class or string reference
            optimizer_kwargs: Dict of optimizer parameters (lr, weight_decay, etc.)
            lr_scheduler_cls: Optional scheduler class or string reference
            lr_scheduler_kwargs: Dict of scheduler parameters
            n_iterations: Number of training iterations
            log_interval: How often to log metrics
            save_interval: Interval (in iterations) at which to save intermediate models (in memory). If None, do not save.
        """
        self.feature_generator = feature_generator
        self.n_batch = n_batch
        self.loss = loss

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.n_iterations = n_iterations
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.hook = hook
        self.saved_models = []
        self.epsilon_early_stopping = epsilon_early_stopping
        self.n_patience = n_patience
    
    def train(self, model):
        """
        Train the model using the configured parameters.
        
        Args:
            model: The model to train (must have n_instances attribute and device)
            feature_generator: Generator function that yields training batches
            importances: Feature importance weights
            
        Returns:
            Self (for method chaining, training history stored in trainer)
        """
        # Keep reference to the model for other methods
        self.model = model

        # Use model's device
        device = next(model.parameters()).device
        self.loss = self.loss.to(device)
        
        # Create optimizer and scheduler with model parameters
        optimizer = ICfg.instantiate_class(self.optimizer_cls, model.parameters(), **self.optimizer_kwargs)
        scheduler = (ICfg.instantiate_class(self.lr_scheduler_cls, optimizer, **self.lr_scheduler_kwargs) 
                    if self.lr_scheduler_cls is not None else None)
        
        # Reset training history
        self.loss_history = []
        self.weight_norms = []
        self.gradient_norms = []
        
        # Training loop
        patience_counter = 0
        prev_avg = None
        for step_idx in tqdm.tqdm(range(self.n_iterations)):
            self.step_idx = step_idx
            # Get batch and prepare for stacked model
            x = self.feature_generator(n_batch=self.n_batch, 
                                  n_features=model.n_features, 
                                  n_instances=model.n_instances, 
                                  device=device)
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            y = model(x)
            loss = self.loss(x, y)
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            self.loss_history.append(loss.item())
            
            # Log metrics periodically
            if self.log_interval is not None and step_idx % self.log_interval == 0:
                weight_norm = torch.norm(model.layers['L_down'].weight).item()
                self.weight_norms.append(weight_norm)
                
                grad_norm = torch.norm(torch.cat([
                    p.grad.flatten() for p in model.parameters() if p.grad is not None
                ])).item()
                self.gradient_norms.append(grad_norm)
            
            if self.save_interval is not None and self.save_interval > 0 and (step_idx % self.save_interval == 0):
                # Save a deepcopy of the model at the specified interval
                import copy
                self.saved_models.append(copy.deepcopy(model))
            
            if self.hook is not None:
                self.hook(model=model, step=step_idx, trainer=self)
            # Early stopping logic
            if self.epsilon_early_stopping is not None and self.n_patience is not None and step_idx >= 50:
                avg = sum(self.loss_history[-50:]) / 50
                if prev_avg is not None and abs(avg - prev_avg) < self.epsilon_early_stopping:
                    patience_counter += 1
                else:
                    patience_counter = 0
                prev_avg = avg
                if patience_counter >= self.n_patience:
                    print(f"Early stopping at step {step_idx}")
                    break
        return self
    
    def best_instance_index(self):  #! untested
        """Get the index of the best instance based on final losses."""
        if not hasattr(self, "model"):
            raise AttributeError("Trainer.train must be called before best_instance_index")

        device = next(self.model.parameters()).device
        batch = self.feature_generator(
            n_batch=self.n_batch * 10,
            n_features=self.model.n_features,
            n_instances=self.model.n_instances,
            device=device,
        )
        final_losses = self.loss(batch, self.model(batch))
        i_min_loss = torch.argmin(final_losses)
        return i_min_loss.item()


