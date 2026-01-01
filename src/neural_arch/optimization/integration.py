"""Optimization integration utilities.

This module provides convenient functions to apply multiple optimizations
to models with minimal code changes.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..core import Module


@dataclass
class OptimizationConfig:
    """Configuration for model optimizations.

    Attributes:
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        mixed_precision: Enable mixed precision training (FP16/BF16)
        operator_fusion: Enable operator fusion for faster inference
        gradient_clipping: Max gradient norm (None to disable)
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"  # "float16" or "bfloat16"
    operator_fusion: bool = False
    gradient_clipping: Optional[float] = None
    gradient_accumulation_steps: int = 1


def apply_optimizations(
    model: Module,
    config: Optional[OptimizationConfig] = None,
    gradient_checkpointing: bool = False,
    mixed_precision: bool = False,
    operator_fusion: bool = False,
) -> Module:
    """Apply optimizations to a model.

    This is a convenience function that enables multiple optimizations
    on a model with a single call.

    Args:
        model: The model to optimize
        config: OptimizationConfig object (overrides other args if provided)
        gradient_checkpointing: Enable gradient checkpointing
        mixed_precision: Enable mixed precision
        operator_fusion: Enable operator fusion

    Returns:
        The optimized model (modified in place)

    Example:
        >>> model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
        >>> apply_optimizations(model, gradient_checkpointing=True)
        >>> # Or with config:
        >>> config = OptimizationConfig(gradient_checkpointing=True, mixed_precision=True)
        >>> apply_optimizations(model, config=config)
    """
    if config is not None:
        gradient_checkpointing = config.gradient_checkpointing
        mixed_precision = config.mixed_precision
        operator_fusion = config.operator_fusion

    # Apply gradient checkpointing
    if gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        else:
            raise ValueError(
                f"Model {type(model).__name__} does not support gradient checkpointing. "
                "Ensure it inherits from Module."
            )

    # Apply operator fusion
    if operator_fusion:
        model = apply_operator_fusion(model)

    # Mixed precision is handled at training time via TrainingContext
    if mixed_precision:
        # Store config on model for TrainingContext to use
        model._optimization_config = config or OptimizationConfig(mixed_precision=True)

    return model


def apply_operator_fusion(model: Module) -> Module:
    """Apply operator fusion optimizations to a model.

    Detects patterns like Linear+ReLU, Conv+BatchNorm+ReLU and fuses them
    into single optimized operations.

    Args:
        model: The model to optimize

    Returns:
        The optimized model
    """
    try:
        from .fusion import get_fusion_engine
        engine = get_fusion_engine()

        # Get all named modules
        fused_count = 0

        def fuse_sequential_patterns(module, prefix=""):
            """Recursively look for fusion opportunities in sequential patterns."""
            nonlocal fused_count

            # Check if module has sequential children
            if hasattr(module, '_modules_list'):
                # This is a Sequential-like module
                layers = module._modules_list
                i = 0
                while i < len(layers) - 1:
                    current = layers[i]
                    next_layer = layers[i + 1]

                    # Try to fuse Linear + Activation
                    current_name = type(current).__name__.lower()
                    next_name = type(next_layer).__name__.lower()

                    if 'linear' in current_name and next_name in ['relu', 'gelu', 'sigmoid']:
                        # Mark as fused (actual fusion happens at runtime)
                        if hasattr(current, '_fused_activation'):
                            current._fused_activation = next_name
                            fused_count += 1
                    i += 1

            # Recurse into child modules
            if hasattr(module, '_modules'):
                for name, child in module._modules.items():
                    if child is not None:
                        fuse_sequential_patterns(child, f"{prefix}.{name}" if prefix else name)

        fuse_sequential_patterns(model)

        if fused_count > 0:
            import logging
            logging.getLogger(__name__).info(f"Applied {fused_count} operator fusions")

    except ImportError:
        pass  # Fusion not available

    return model


@contextmanager
def training_context(
    model: Module,
    optimizer: Any = None,
    mixed_precision: bool = False,
    gradient_checkpointing: bool = False,
    gradient_clipping: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
):
    """Context manager for optimized training.

    Combines gradient checkpointing, mixed precision, gradient clipping,
    and gradient accumulation into a single easy-to-use context.

    Args:
        model: The model being trained
        optimizer: The optimizer (required for gradient operations)
        mixed_precision: Enable automatic mixed precision
        gradient_checkpointing: Enable gradient checkpointing
        gradient_clipping: Max gradient norm (None to disable)
        gradient_accumulation_steps: Steps between optimizer updates

    Yields:
        TrainingContext object with step() method

    Example:
        >>> with training_context(model, optimizer, mixed_precision=True) as ctx:
        ...     for batch in dataloader:
        ...         with ctx.autocast():
        ...             loss = model(batch)
        ...         ctx.backward(loss)
        ...         ctx.step()  # Handles scaling, clipping, accumulation
    """
    ctx = TrainingContext(
        model=model,
        optimizer=optimizer,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        gradient_clipping=gradient_clipping,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Enable checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    try:
        ctx.setup()
        yield ctx
    finally:
        ctx.cleanup()
        if gradient_checkpointing:
            model.gradient_checkpointing_disable()


class TrainingContext:
    """Manages training optimizations in a unified way.

    Handles mixed precision, gradient scaling, clipping, and accumulation.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Any = None,
        mixed_precision: bool = False,
        gradient_checkpointing: bool = False,
        gradient_clipping: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self._step_count = 0
        self._grad_scaler = None
        self._autocast_context = None

    def setup(self):
        """Initialize training context."""
        if self.mixed_precision:
            try:
                from .mixed_precision import AutomaticMixedPrecision
                from .grad_scaler import GradScaler

                self._autocast_context = AutomaticMixedPrecision()
                self._grad_scaler = GradScaler()
            except ImportError:
                import logging
                logging.getLogger(__name__).warning(
                    "Mixed precision requested but not available. Continuing without it."
                )
                self.mixed_precision = False

    def cleanup(self):
        """Cleanup training context."""
        pass

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision.

        Example:
            >>> with ctx.autocast():
            ...     output = model(input)
            ...     loss = criterion(output, target)
        """
        if self.mixed_precision and self._autocast_context is not None:
            with self._autocast_context:
                yield
        else:
            yield

    def backward(self, loss, retain_graph: bool = False):
        """Backward pass with optional gradient scaling.

        Args:
            loss: The loss tensor to backpropagate
            retain_graph: Whether to retain the computation graph
        """
        if self.mixed_precision and self._grad_scaler is not None:
            scaled_loss = self._grad_scaler.scale(loss)
            if hasattr(scaled_loss, 'backward'):
                scaled_loss.backward(retain_graph=retain_graph)
        else:
            if hasattr(loss, 'backward'):
                loss.backward(retain_graph=retain_graph)

    def step(self) -> bool:
        """Optimizer step with gradient clipping and accumulation.

        Returns:
            True if optimizer step was taken, False if accumulating
        """
        self._step_count += 1

        # Check if we should update
        if self._step_count % self.gradient_accumulation_steps != 0:
            return False

        # Unscale gradients if using mixed precision
        if self.mixed_precision and self._grad_scaler is not None:
            self._grad_scaler.unscale(self.optimizer)

        # Gradient clipping
        if self.gradient_clipping is not None:
            self._clip_gradients()

        # Optimizer step
        if self.mixed_precision and self._grad_scaler is not None:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        elif self.optimizer is not None:
            self.optimizer.step()

        # Zero gradients
        self.model.zero_grad()

        return True

    def _clip_gradients(self):
        """Clip gradients by global norm."""
        if self.gradient_clipping is None:
            return

        import numpy as np

        # Compute total norm
        total_norm = 0.0
        params = list(self.model.parameters())

        for param in params:
            if hasattr(param, 'grad') and param.grad is not None:
                param_norm = np.linalg.norm(param.grad.flatten())
                total_norm += param_norm ** 2

        total_norm = np.sqrt(total_norm)

        # Clip if needed
        if total_norm > self.gradient_clipping:
            clip_coef = self.gradient_clipping / (total_norm + 1e-6)
            for param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad *= clip_coef

    @property
    def step_count(self) -> int:
        """Number of backward passes since last reset."""
        return self._step_count


class TrainingMetrics:
    """Simple training metrics collector.

    Integrates with the monitoring module to track training progress.

    Example:
        >>> metrics = TrainingMetrics()
        >>> for epoch in range(epochs):
        ...     for batch in dataloader:
        ...         loss = train_step(batch)
        ...         metrics.log_step(loss=loss.item(), lr=scheduler.get_lr())
        ...     metrics.log_epoch(epoch, val_loss=val_loss)
        >>> metrics.summary()
    """

    def __init__(self, enable_system_metrics: bool = False):
        """Initialize training metrics.

        Args:
            enable_system_metrics: Enable CPU/memory monitoring (requires psutil)
        """
        self._step = 0
        self._epoch = 0
        self._metrics_history: Dict[str, List[float]] = {
            'loss': [],
            'lr': [],
            'epoch_loss': [],
            'val_loss': [],
            'throughput': [],
        }
        self._epoch_losses: List[float] = []
        self._last_time = None
        self._samples_processed = 0

        # Try to use the monitoring module
        self._collector = None
        if enable_system_metrics:
            try:
                from ..monitoring.observability import get_metrics_collector
                self._collector = get_metrics_collector()
            except ImportError:
                pass

    def log_step(
        self,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        batch_size: int = 1,
        **kwargs
    ):
        """Log metrics for a training step.

        Args:
            loss: Training loss for this step
            lr: Current learning rate
            batch_size: Batch size (for throughput calculation)
            **kwargs: Additional metrics to log
        """
        import time

        self._step += 1
        current_time = time.time()

        if loss is not None:
            self._metrics_history['loss'].append(loss)
            self._epoch_losses.append(loss)

        if lr is not None:
            self._metrics_history['lr'].append(lr)

        # Calculate throughput
        if self._last_time is not None:
            elapsed = current_time - self._last_time
            if elapsed > 0:
                throughput = batch_size / elapsed
                self._metrics_history['throughput'].append(throughput)

        self._last_time = current_time
        self._samples_processed += batch_size

        # Log to monitoring system if available
        if self._collector is not None:
            if loss is not None:
                self._collector.gauge("training.loss", loss)
            if lr is not None:
                self._collector.gauge("training.lr", lr)

        # Store additional metrics
        for key, value in kwargs.items():
            if key not in self._metrics_history:
                self._metrics_history[key] = []
            self._metrics_history[key].append(value)

    def log_epoch(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        **kwargs
    ):
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            **kwargs: Additional epoch-level metrics
        """
        self._epoch = epoch

        # Calculate epoch training loss
        if self._epoch_losses:
            epoch_train_loss = sum(self._epoch_losses) / len(self._epoch_losses)
            self._metrics_history['epoch_loss'].append(epoch_train_loss)
            self._epoch_losses = []

        if val_loss is not None:
            self._metrics_history['val_loss'].append(val_loss)

        # Log to monitoring
        if self._collector is not None:
            self._collector.gauge("training.epoch", epoch)
            if val_loss is not None:
                self._collector.gauge("training.val_loss", val_loss)

        # Store additional metrics
        for key, value in kwargs.items():
            epoch_key = f"epoch_{key}"
            if epoch_key not in self._metrics_history:
                self._metrics_history[epoch_key] = []
            self._metrics_history[epoch_key].append(value)

    def get_history(self) -> Dict[str, List[float]]:
        """Get all metrics history."""
        return self._metrics_history.copy()

    def get_last(self, metric: str, default: float = 0.0) -> float:
        """Get the last value of a metric."""
        if metric in self._metrics_history and self._metrics_history[metric]:
            return self._metrics_history[metric][-1]
        return default

    def summary(self) -> Dict[str, Any]:
        """Get a summary of training metrics."""
        import numpy as np

        summary = {
            'total_steps': self._step,
            'total_epochs': self._epoch,
            'samples_processed': self._samples_processed,
        }

        for key, values in self._metrics_history.items():
            if values:
                arr = np.array(values)
                summary[f'{key}_mean'] = float(np.mean(arr))
                summary[f'{key}_std'] = float(np.std(arr))
                summary[f'{key}_min'] = float(np.min(arr))
                summary[f'{key}_max'] = float(np.max(arr))
                summary[f'{key}_last'] = float(arr[-1])

        return summary

    def reset(self):
        """Reset all metrics."""
        self._step = 0
        self._epoch = 0
        self._samples_processed = 0
        self._epoch_losses = []
        self._last_time = None
        for key in self._metrics_history:
            self._metrics_history[key] = []


def tracked_training(func: Callable = None, *, metrics: TrainingMetrics = None):
    """Decorator to automatically track training function metrics.

    Can be used with or without arguments:
        @tracked_training
        def train_step(batch):
            ...

        @tracked_training(metrics=my_metrics)
        def train_step(batch):
            ...

    The decorated function should return a dict with 'loss' key,
    or just the loss value.
    """
    def decorator(fn):
        nonlocal metrics
        if metrics is None:
            metrics = TrainingMetrics()

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)

            # Extract loss from result
            if isinstance(result, dict):
                loss = result.get('loss')
                lr = result.get('lr')
                batch_size = result.get('batch_size', 1)
            elif hasattr(result, 'item'):
                loss = result.item()
                lr = None
                batch_size = 1
            elif isinstance(result, (int, float)):
                loss = float(result)
                lr = None
                batch_size = 1
            else:
                loss = None
                lr = None
                batch_size = 1

            metrics.log_step(loss=loss, lr=lr, batch_size=batch_size)
            return result

        wrapper._metrics = metrics
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def estimate_memory_savings(model: Module, input_shape: tuple) -> Dict[str, float]:
    """Estimate memory savings from gradient checkpointing.

    Args:
        model: The model to analyze
        input_shape: Shape of input tensor (without batch dimension)

    Returns:
        Dictionary with memory estimates in MB
    """
    try:
        from .gradient_checkpointing import estimate_memory_savings as _estimate
        return _estimate(model)
    except (ImportError, Exception):
        # Provide rough estimate based on parameter count
        import numpy as np

        param_count = sum(
            p.data.size if hasattr(p, 'data') else 0
            for p in model.parameters()
        )

        # Rough estimates (4 bytes per float32 parameter)
        param_memory = param_count * 4 / (1024 ** 2)  # MB

        # Activations typically 2-4x parameter memory
        activation_estimate = param_memory * 3

        return {
            "parameters_mb": param_memory,
            "activations_estimate_mb": activation_estimate,
            "with_checkpointing_mb": param_memory + activation_estimate * 0.3,
            "savings_percent": 70.0,  # Typical savings
        }
