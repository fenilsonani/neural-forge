"""Neural network optimization and acceleration utilities.

This package provides enterprise-grade optimization tools including:
- Operator fusion for 2-5x performance improvements
- Mixed precision training with automatic loss scaling
- JIT compilation and kernel optimization
- Memory optimization and gradient checkpointing
"""

from .fusion import (
    FusionEngine,
    fuse_conv_bn_activation,
    fuse_layernorm_linear,
    fuse_linear_activation,
    get_fusion_engine,
)
from .mixed_precision import (
    AutomaticMixedPrecision,
    GradScaler,
    MixedPrecisionManager,
    PrecisionConfig,
    AutocastConfig,
    AutocastPolicy,
    get_mixed_precision_manager,
    autocast,
    create_precision_config,
    get_recommended_precision_config,
    create_training_context,
)

# Import advanced gradient scaler if available
try:
    from .grad_scaler import (
        AdvancedGradScaler,
        ScalerConfig,
        ScalingStrategy,
        create_scaler,
        check_gradients_finite,
        clip_gradients_by_norm,
    )
    _ADVANCED_SCALER_AVAILABLE = True
except ImportError:
    _ADVANCED_SCALER_AVAILABLE = False

# Import gradient checkpointing
try:
    from .gradient_checkpointing import (
        CheckpointFunction,
        GradientCheckpointManager,
        get_checkpoint_manager,
        checkpoint_scope,
        checkpoint,
        SequentialCheckpoint,
        memory_efficient_attention,
        CheckpointedTransformerLayer,
        estimate_memory_savings,
        checkpoint_sequential,
        no_checkpoint,
        force_checkpoint,
    )
    _GRADIENT_CHECKPOINTING_AVAILABLE = True
except ImportError:
    _GRADIENT_CHECKPOINTING_AVAILABLE = False

# Import AMP optimizer if available
try:
    from .amp_optimizer import (
        AMPOptimizer,
        AMPOptimizerFactory,
        AMPContext,
        create_amp_adam,
        create_amp_adamw,
        create_amp_sgd,
        get_recommended_scaler_config,
    )
    _AMP_OPTIMIZER_AVAILABLE = True
except ImportError:
    _AMP_OPTIMIZER_AVAILABLE = False

__all__ = [
    # Operator fusion
    "FusionEngine",
    "get_fusion_engine",
    "fuse_linear_activation",
    "fuse_conv_bn_activation",
    "fuse_layernorm_linear",
    # Mixed precision training - core
    "MixedPrecisionManager",
    "AutomaticMixedPrecision",
    "GradScaler",
    "get_mixed_precision_manager",
    "autocast",
    # Mixed precision training - configuration
    "PrecisionConfig",
    "AutocastConfig",
    "AutocastPolicy",
    "create_precision_config",
    "get_recommended_precision_config",
    "create_training_context",
]

# Add advanced scaler exports if available
if _ADVANCED_SCALER_AVAILABLE:
    __all__.extend([
        "AdvancedGradScaler",
        "ScalerConfig",
        "ScalingStrategy",
        "create_scaler",
        "check_gradients_finite",
        "clip_gradients_by_norm",
    ])

# Add AMP optimizer exports if available
if _AMP_OPTIMIZER_AVAILABLE:
    __all__.extend([
        "AMPOptimizer",
        "AMPOptimizerFactory",
        "AMPContext",
        "create_amp_adam",
        "create_amp_adamw",
        "create_amp_sgd",
        "get_recommended_scaler_config",
    ])

# Add gradient checkpointing exports if available
if _GRADIENT_CHECKPOINTING_AVAILABLE:
    __all__.extend([
        "CheckpointFunction",
        "GradientCheckpointManager",
        "get_checkpoint_manager",
        "checkpoint_scope",
        "checkpoint",
        "SequentialCheckpoint",
        "memory_efficient_attention",
        "CheckpointedTransformerLayer",
        "estimate_memory_savings",
        "checkpoint_sequential",
        "no_checkpoint",
        "force_checkpoint",
    ])

# Import integration utilities
try:
    from .integration import (
        OptimizationConfig,
        apply_optimizations,
        apply_operator_fusion,
        training_context,
        TrainingContext,
        TrainingMetrics,
        tracked_training,
        estimate_memory_savings as estimate_optimization_savings,
    )
    _INTEGRATION_AVAILABLE = True
except Exception as _e:
    import logging as _logging
    _logging.getLogger(__name__).debug(f"Integration import failed: {_e}")
    _INTEGRATION_AVAILABLE = False

# Add integration exports if available
if _INTEGRATION_AVAILABLE:
    __all__.extend([
        "OptimizationConfig",
        "apply_optimizations",
        "apply_operator_fusion",
        "training_context",
        "TrainingContext",
        "TrainingMetrics",
        "tracked_training",
    ])
