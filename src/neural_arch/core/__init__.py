"""Core neural architecture components."""

from typing import Optional, Union, Any, List, Tuple
import numpy as np

# Type aliases
TensorLike = Union['Tensor', np.ndarray, list, float, int]
Shape = Tuple[int, ...]


class GradientFunction:
    """Base class for gradient functions."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


from enum import Enum


class DeviceType(Enum):
    """Device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Device:
    """Device abstraction for compute location (CPU/GPU)."""

    def __init__(self, device_type: str = "cpu"):
        """Initialize device.

        Args:
            device_type: Type of device ('cpu', 'cuda', 'mps')
        """
        self.type = device_type.lower()
        if self.type not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unknown device type: {device_type}")

    def __str__(self):
        return f"Device({self.type})"

    def __repr__(self):
        return self.__str__()


class DType:
    """Data type abstraction."""

    FLOAT32 = np.float32
    FLOAT64 = np.float64
    INT32 = np.int32
    INT64 = np.int64
    BOOL = np.bool_

    def __init__(self, dtype):
        """Initialize data type."""
        self.dtype = dtype

    def __str__(self):
        return str(self.dtype)


class Tensor:
    """Basic tensor implementation."""

    def __init__(self, data, dtype=None, device=None, requires_grad=False, name=None):
        """Initialize tensor.

        Args:
            data: Input data (list, numpy array, or scalar)
            dtype: Data type
            device: Compute device
            requires_grad: Whether to track gradients
            name: Optional tensor name
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        if dtype is not None:
            self.data = self.data.astype(dtype)

        self.device = device or Device("cpu")
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.name = name

        # Backend compatibility
        from ..backends.backend import Backend
        self.backend = Backend.get_backend(self.device.type)
        self.backend_data = self.data  # For now, just use numpy

    def __str__(self):
        return f"Tensor({self.data})"

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def backward(self):
        """Compute gradients."""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradients")
        # Simplified backward pass
        if self.grad is None:
            self.grad = np.ones_like(self.data)


class Parameter(Tensor):
    """Parameter is a special tensor used for model weights."""

    def __init__(self, data, dtype=None, device=None):
        """Initialize parameter (always requires gradients)."""
        super().__init__(data, dtype=dtype, device=device, requires_grad=True)


class Optimizer:
    """Base optimizer class."""

    def __init__(self, params, lr=0.01):
        """Initialize optimizer."""
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Perform optimization step."""
        pass

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad = None


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        """Initialize module."""
        self.training = True
        self._parameters = {}
        self._modules = {}

    def forward(self, *args, **kwargs):
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Get all parameters."""
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


# Global gradient tracking state
_grad_enabled = True


def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    return _grad_enabled


def enable_grad():
    """Enable gradient computation."""
    global _grad_enabled
    _grad_enabled = True


class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        """Enter context."""
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        """Exit context."""
        global _grad_enabled
        _grad_enabled = self.prev


# Default device and dtype
_default_device = Device("cpu")
_default_dtype = DType.FLOAT32


def get_default_device() -> Device:
    """Get default device."""
    return _default_device


def set_default_device(device: Union[str, Device]):
    """Set default device."""
    global _default_device
    if isinstance(device, str):
        device = Device(device)
    _default_device = device


def get_default_dtype():
    """Get default data type."""
    return _default_dtype


def set_default_dtype(dtype):
    """Set default data type."""
    global _default_dtype
    _default_dtype = dtype


# Export all components
__all__ = [
    'Device',
    'DType',
    'Tensor',
    'Parameter',
    'Module',
    'is_grad_enabled',
    'enable_grad',
    'no_grad',
    'get_default_device',
    'set_default_device',
    'get_default_dtype',
    'set_default_dtype',
    'GradientFunction',
    'TensorLike',
    'Shape',
    'DeviceType',
    'Optimizer',
]