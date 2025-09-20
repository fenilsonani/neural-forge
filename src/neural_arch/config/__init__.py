"""Configuration management for neural architecture."""

from .defaults import Config, DEFAULT_CONFIG, get_preset_config

# Create dummy functions to match expected interface
def ConfigManager():
    return DEFAULT_CONFIG

def load_config(path=None):
    return DEFAULT_CONFIG

def save_config(config, path):
    pass
from .defaults import (
    DEFAULT_CONFIG,
    DEVELOPMENT_CONFIG,
    PRODUCTION_CONFIG,
    get_preset_config,
    list_preset_configs,
)
from .validation import ConfigValidator, validate_config

__all__ = [
    "Config",
    "ConfigManager",
    "load_config",
    "save_config",
    "get_preset_config",
    "list_preset_configs",
    "DEFAULT_CONFIG",
    "PRODUCTION_CONFIG",
    "DEVELOPMENT_CONFIG",
    "ConfigValidator",
    "validate_config",
]
