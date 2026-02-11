"""Manhua Image Cleaner - AI-powered text removal from manga/manhua images."""

__version__ = "2.0.0"

from .config import ModelType, Backend
from .core import BatchProcessor, ProcessingConfig, ProcessingResult
from .exceptions import (
    ManhuaCleanerError,
    ConfigurationError,
    ImageProcessingError,
    OCRError,
    ModelError,
    ValidationError,
    WorkerError,
)
from .utils.env import setup_logging

__all__ = [
    '__version__',
    'ModelType',
    'Backend',
    'ProcessingConfig',
    'BatchProcessor',
    'ProcessingResult',
    'setup_logging',
    # Exceptions
    'ManhuaCleanerError',
    'ConfigurationError',
    'ImageProcessingError',
    'OCRError',
    'ModelError',
    'ValidationError',
    'WorkerError',
]
