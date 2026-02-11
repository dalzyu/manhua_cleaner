"""Custom exceptions for Manhua Image Cleaner."""

from typing import Optional


class ManhuaCleanerError(Exception):
    """Base exception for all application errors.
    
    Attributes:
        message: Human-readable error description
        error_code: Optional error code for programmatic handling
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(ManhuaCleanerError):
    """Error in configuration or settings.
    
    Attributes:
        config_key: The configuration key that caused the error (if applicable)
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, error_code="CONFIG_ERROR")
        self.config_key = config_key


class ImageProcessingError(ManhuaCleanerError):
    """Error during image processing.
    
    Attributes:
        image_path: Path to the image being processed when error occurred
    """
    
    def __init__(self, message: str, image_path: Optional[str] = None):
        super().__init__(message, error_code="IMAGE_ERROR")
        self.image_path = image_path
    
    def __str__(self) -> str:
        if self.image_path:
            return f"{super().__str__()} (image: {self.image_path})"
        return super().__str__()


class OCRError(ManhuaCleanerError):
    """Error during OCR/text detection.
    
    Attributes:
        image_path: Path to the image being processed when error occurred
    """
    
    def __init__(self, message: str, image_path: Optional[str] = None):
        super().__init__(message, error_code="OCR_ERROR")
        self.image_path = image_path


class ModelError(ManhuaCleanerError):
    """Error loading or running AI model.
    
    Attributes:
        model_id: The model ID that caused the error (if applicable)
    """
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(message, error_code="MODEL_ERROR")
        self.model_id = model_id


class ValidationError(ManhuaCleanerError):
    """Error validating inputs or parameters.
    
    Attributes:
        field: The field that failed validation (if applicable)
    """
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field


class WorkerError(ManhuaCleanerError):
    """Error in multiprocessing worker.
    
    Attributes:
        worker_id: The worker ID that encountered the error (if applicable)
    """
    
    def __init__(self, message: str, worker_id: Optional[int] = None):
        super().__init__(message, error_code="WORKER_ERROR")
        self.worker_id = worker_id
