"""Neural Architecture - Enterprise-Grade Neural Network Implementation.

A comprehensive neural network framework built from scratch with NumPy,
featuring enterprise-grade architecture, comprehensive testing, and
production-ready performance.

Key Features:
- Custom tensor system with automatic differentiation
- Complete neural network layers and optimizers
- Transformer architecture with attention mechanisms
- Enterprise-grade error handling and logging
- Configuration management and CLI tools
- Comprehensive test suite and benchmarking
"""

# Version information
from .__version__ import __version__, __version_info__

# Configuration and utilities
from .config import Config, get_preset_config, load_config, save_config

# Core components
from .core import (
    Device,
    DType,
    Module,
    Parameter,
    Tensor,
    enable_grad,
    get_default_device,
    get_default_dtype,
    is_grad_enabled,
    no_grad,
    set_default_device,
    set_default_dtype,
)

# Exceptions
from .exceptions import (
    DeviceError,
    DTypeError,
    GradientError,
    NeuralArchError,
    ShapeError,
    TensorError,
)

# Functional operations
from .functional import (
    add,
    cross_entropy_loss,
    div,
    matmul,
    max_pool,
    mean_pool,
    mse_loss,
    mul,
    neg,
    relu,
    sigmoid,
    softmax,
    sub,
    tanh,
)

# Neural network layers
from .nn import (
    GELU,
    Embedding,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    TransformerBlock,
    # Modern attention mechanisms
    DifferentialAttention,
    DifferentialTransformer,
    GroupedQueryAttention,
    MultiQueryAttention,
)

# Optimizers
from .optim import SGD, Adam, AdamW

# Import optimization configuration system
try:
    from .optimization_config import configure
    from .optimization_config import get_config as get_optimization_config
    from .optimization_config import reset_config

    _optimization_config_available = True
except ImportError:
    # Optimization config not available
    _optimization_config_available = False

# Import gradient checkpointing
try:
    from .optimization import (
        checkpoint,
        checkpoint_sequential,
        get_checkpoint_manager,
    )
    _gradient_checkpointing_available = True
except ImportError:
    _gradient_checkpointing_available = False

# Import visualization module
try:
    from .visualization import (
        # Architecture visualization
        ModelVisualizer,
        plot_model_architecture,
        plot_computational_graph,
        # Training visualization
        TrainingVisualizer,
        plot_training_curves,
        plot_loss_history,
        # Feature visualization
        FeatureVisualizer,
        plot_feature_maps,
        plot_activations,
        visualize_attention_weights,
        # Performance visualization
        PerformanceVisualizer,
        plot_benchmark_results,
        # Interactive visualization
        InteractiveVisualizer,
        create_streamlit_dashboard,
    )
    _visualization_available = True
except ImportError:
    _visualization_available = False

# CLI
from .cli import main as cli_main

# Backward compatibility utilities
from .utils import create_text_vocab, propagate_gradients, text_to_sequences

# Set up enterprise-grade public API
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    # Core tensor system
    "Tensor",
    "Parameter",
    "Module",
    "Device",
    "DType",
    # Device management
    "get_default_device",
    "set_default_device",
    "get_default_dtype",
    "set_default_dtype",
    # Gradient control
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    # Functional operations
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "matmul",
    "relu",
    "softmax",
    "sigmoid",
    "tanh",
    "mean_pool",
    "max_pool",
    "cross_entropy_loss",
    "mse_loss",
    # Neural network layers
    "Linear",
    "Embedding",
    "LayerNorm",
    "ReLU",
    "Softmax",
    "Sigmoid",
    "Tanh",
    "GELU",
    "MultiHeadAttention",
    "TransformerBlock",
    # Modern attention mechanisms
    "DifferentialAttention",
    "DifferentialTransformer",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    # Gradient checkpointing
    "checkpoint",
    "checkpoint_sequential",
    "get_checkpoint_manager",
    # Visualization
    "ModelVisualizer",
    "plot_model_architecture",
    "plot_computational_graph",
    "TrainingVisualizer",
    "plot_training_curves",
    "plot_loss_history",
    "FeatureVisualizer",
    "plot_feature_maps",
    "plot_activations",
    "visualize_attention_weights",
    "PerformanceVisualizer",
    "plot_benchmark_results",
    "InteractiveVisualizer",
    "create_streamlit_dashboard",
    # Optimizers
    "Adam",
    "SGD",
    "AdamW",
    # Configuration
    "Config",
    "load_config",
    "save_config",
    "get_preset_config",
    # Exceptions
    "NeuralArchError",
    "TensorError",
    "ShapeError",
    "DTypeError",
    "DeviceError",
    "GradientError",
    # Backward compatibility
    "propagate_gradients",
    "create_text_vocab",
    "text_to_sequences",
]

# Add optimization config if available
if _optimization_config_available:
    __all__.extend(["configure", "get_optimization_config", "reset_config"])

# Enterprise components (optional imports for advanced users)
try:
    from .backends.cuda_kernels_optimized import OptimizedCUDAKernels
    from .core.autograd import GradientTape
    from .core.memory_manager import AdvancedMemoryManager
    from .monitoring.observability import MetricsCollector, DistributedTracer
    from .optimization.advanced_optimizers import SophiaOptimizer, LionOptimizer
    
    # Enterprise components available
    _enterprise_available = True
    
    # Add to public API
    __all__.extend([
        # CUDA Optimization
        "OptimizedCUDAKernels",
        # Advanced Autograd
        "GradientTape",
        # Memory Management
        "AdvancedMemoryManager",
        # Monitoring
        "MetricsCollector",
        "DistributedTracer",
        # Advanced Optimizers
        "SophiaOptimizer",
        "LionOptimizer",
    ])
    
except ImportError as e:
    # Enterprise components not available
    _enterprise_available = False

# Enterprise features configuration
import logging

# Set up default logging (can be overridden by user)
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Only add handler if none exists (avoid duplicate logs)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Default to WARNING level


def run_cli(*args):
    """Run the command-line interface programmatically.

    Args:
        *args: Command line arguments

    Returns:
        Exit code
    """
    return cli_main(list(args))


# Add CLI to public API
__all__.append("run_cli")


def is_enterprise_available():
    """Check if enterprise components are available.
    
    Returns:
        bool: True if all enterprise components are available
    """
    return _enterprise_available


def list_enterprise_components():
    """List available enterprise components.
    
    Returns:
        dict: Dictionary of component categories and their classes
    """
    if not _enterprise_available:
        return {}
    
    return {
        "cuda_optimization": ["OptimizedCUDAKernels"],
        "advanced_autograd": ["GradientTape"],
        "memory_management": ["AdvancedMemoryManager"],
        "monitoring": ["MetricsCollector", "DistributedTracer"],
        "advanced_optimizers": ["SophiaOptimizer", "LionOptimizer"],
    }


# Add enterprise utility functions to public API
__all__.extend(["is_enterprise_available", "list_enterprise_components"])
