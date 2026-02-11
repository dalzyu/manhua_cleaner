"""Settings persistence manager for the GUI."""

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSettings

from ..config import ModelType, Backend

logger = logging.getLogger(__name__)


# Default values for all settings
DEFAULT_SETTINGS = {
    # Model params
    "params/model": ModelType.FLUX_2_KLEIN_4B.value,
    "params/backend": Backend.AUTO.value,
    "params/steps": 4,
    "params/expand": 25,
    "params/smart_fill": True,
    "params/smart_fill_expand": 5,
    "params/color_correct": True,
    "params/edge_blend": True,
    "params/prompt": "remove all text",
    # OCR settings
    "ocr/version": "v1.5",
    "ocr/precision": "fp16",
    "ocr/tensorrt": True,
    "ocr/workers": 1,
    # Whitelist settings
    "whitelist/enabled": False,
    "whitelist/preset": "none",
    "whitelist/distance": 50,
    "whitelist/custom": "",
    # Extra pass upscaling settings
    "upscale/enabled": False,
    "upscale/factor": "2x",
    "upscale/method": "lanczos",
}


class SettingsManager:
    """Manages application settings persistence using QSettings.
    
    This class handles loading and saving of all user preferences,
    window geometry, and recent folders.
    """
    
    SETTINGS_FILE = "gui_settings.ini"
    MAX_RECENT_FOLDERS = 5
    
    def __init__(self, parent_widget=None):
        """Initialize settings manager.
        
        Args:
            parent_widget: Parent widget for settings context (optional)
        """
        settings_path = Path(__file__).parent / self.SETTINGS_FILE
        self.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        self._recent_input_folders: list[str] = []
        self._recent_output_folders: list[str] = []
        
    def load_window_geometry(self, window) -> None:
        """Load and apply window geometry and state.
        
        Args:
            window: Main window instance to apply geometry to
        """
        if self.settings.contains("window/geometry"):
            window.restoreGeometry(self.settings.value("window/geometry"))
        if self.settings.contains("window/state"):
            window.restoreState(self.settings.value("window/state"))
    
    def save_window_geometry(self, window) -> None:
        """Save window geometry and state.
        
        Args:
            window: Main window instance to save geometry from
        """
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.setValue("window/state", window.saveState())
    
    def load_recent_folders(self) -> None:
        """Load recent input/output folders lists."""
        self._recent_input_folders = self.settings.value("recent/input_folders", [])
        self._recent_output_folders = self.settings.value("recent/output_folders", [])
        
        # Ensure lists
        if not isinstance(self._recent_input_folders, list):
            self._recent_input_folders = []
        if not isinstance(self._recent_output_folders, list):
            self._recent_output_folders = []
    
    def save_recent_folders(self) -> None:
        """Save recent input/output folders lists."""
        self.settings.setValue("recent/input_folders", self._recent_input_folders)
        self.settings.setValue("recent/output_folders", self._recent_output_folders)
    
    def add_recent_input_folder(self, path: str) -> None:
        """Add a folder to recent input folders list.
        
        Args:
            path: Folder path to add
        """
        self._recent_input_folders = [p for p in self._recent_input_folders if p != path]
        self._recent_input_folders.insert(0, path)
        self._recent_input_folders = self._recent_input_folders[:self.MAX_RECENT_FOLDERS]
    
    def add_recent_output_folder(self, path: str) -> None:
        """Add a folder to recent output folders list.
        
        Args:
            path: Folder path to add
        """
        self._recent_output_folders = [p for p in self._recent_output_folders if p != path]
        self._recent_output_folders.insert(0, path)
        self._recent_output_folders = self._recent_output_folders[:self.MAX_RECENT_FOLDERS]
    
    def get_recent_input_folder(self) -> str:
        """Get most recent input folder (or empty string if none)."""
        return self._recent_input_folders[0] if self._recent_input_folders else ""
    
    def get_recent_output_folder(self) -> str:
        """Get most recent output folder (or empty string if none)."""
        return self._recent_output_folders[0] if self._recent_output_folders else ""
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from settings with proper type handling.
        
        Args:
            key: Settings key
            default: Default value if key not found
            
        Returns:
            Boolean value
        """
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def load_params(self, window) -> None:
        """Load all parameter settings into the window.
        
        Args:
            window: MainWindow instance with UI controls to populate
        """
        # Model
        model = self.settings.value("params/model", ModelType.FLUX_2_KLEIN_4B.value)
        window.model_combo.setCurrentText(model)
        
        # Backend
        backend = self.settings.value("params/backend", Backend.AUTO.value)
        window.backend_combo.setCurrentText(backend)
        
        # Steps
        steps = int(self.settings.value("params/steps", 4))
        window.steps_spin.setValue(steps)
        
        # Expand pixels
        expand = int(self.settings.value("params/expand", 25))
        window.expand_spin.setValue(expand)
        
        # Smart fill
        smart_fill = self._get_bool("params/smart_fill", True)
        window.smart_fill_check.setChecked(smart_fill)
        
        smart_fill_expand = int(self.settings.value("params/smart_fill_expand", 5))
        window.smart_fill_expand_spin.setValue(smart_fill_expand)
        
        # Color correction
        color_correct = self._get_bool("params/color_correct", True)
        window.color_correct_check.setChecked(color_correct)
        
        # Edge blending
        edge_blend = self._get_bool("params/edge_blend", True)
        window.edge_blend_check.setChecked(edge_blend)
        
        # Prompt
        prompt = self.settings.value("params/prompt", "remove all text")
        window.prompt_input.setText(prompt)
        
        # OCR settings
        ocr_version = self.settings.value("ocr/version", "v1.5")
        window.ocr_version_combo.setCurrentText(ocr_version)
        
        ocr_precision = self.settings.value("ocr/precision", "fp16")
        window.ocr_precision_combo.setCurrentText(ocr_precision)
        
        ocr_tensorrt = self._get_bool("ocr/tensorrt", True)
        window.ocr_tensorrt_check.setChecked(ocr_tensorrt)
        
        # OCR Workers
        ocr_workers = int(self.settings.value("ocr/workers", 1))
        window.ocr_workers_spin.setValue(ocr_workers)
        
        # OCR Model
        ocr_model = self.settings.value("ocr/model", "paddleocr")
        window.ocr_model_combo.setCurrentText(ocr_model)
        
        # Whitelist settings
        whitelist_enabled = self._get_bool("whitelist/enabled", False)
        window.whitelist_enable_check.setChecked(whitelist_enabled)
        
        whitelist_preset = self.settings.value("whitelist/preset", "none")
        window.whitelist_preset_combo.setCurrentText(whitelist_preset)
        
        whitelist_distance = int(self.settings.value("whitelist/distance", 50))
        window.whitelist_distance_spin.setValue(whitelist_distance)
        
        whitelist_custom = self.settings.value("whitelist/custom", "")
        window.whitelist_custom_input.setPlainText(whitelist_custom)
        
        # Extra pass upscaling settings
        upscale_enabled = self._get_bool("upscale/enabled", False)
        window.extra_pass_upscale_check.setChecked(upscale_enabled)
        
        upscale_factor = self.settings.value("upscale/factor", "2x")
        window.upscale_factor_combo.setCurrentText(upscale_factor)
        
        upscale_method = self.settings.value("upscale/method", "lanczos")
        window.upscale_method_combo.setCurrentText(upscale_method)
    
    def save_params(self, window) -> None:
        """Save all parameter settings from the window.
        
        Args:
            window: MainWindow instance with UI controls to read from
        """
        self.settings.setValue("params/model", window.model_combo.currentText())
        self.settings.setValue("params/backend", window.backend_combo.currentText())
        self.settings.setValue("params/steps", window.steps_spin.value())
        self.settings.setValue("params/expand", window.expand_spin.value())
        self.settings.setValue("params/smart_fill", window.smart_fill_check.isChecked())
        self.settings.setValue("params/smart_fill_expand", window.smart_fill_expand_spin.value())
        self.settings.setValue("params/color_correct", window.color_correct_check.isChecked())
        self.settings.setValue("params/edge_blend", window.edge_blend_check.isChecked())
        self.settings.setValue("params/prompt", window.prompt_input.text())
        
        # OCR settings
        self.settings.setValue("ocr/model", window.ocr_model_combo.currentText())
        self.settings.setValue("ocr/version", window.ocr_version_combo.currentText())
        self.settings.setValue("ocr/precision", window.ocr_precision_combo.currentText())
        self.settings.setValue("ocr/tensorrt", window.ocr_tensorrt_check.isChecked())
        self.settings.setValue("ocr/workers", window.ocr_workers_spin.value())
        
        # Whitelist settings
        self.settings.setValue("whitelist/enabled", window.whitelist_enable_check.isChecked())
        self.settings.setValue("whitelist/preset", window.whitelist_preset_combo.currentText())
        self.settings.setValue("whitelist/distance", window.whitelist_distance_spin.value())
        self.settings.setValue("whitelist/custom", window.whitelist_custom_input.toPlainText())
        
        # Extra pass upscaling settings
        self.settings.setValue("upscale/enabled", window.extra_pass_upscale_check.isChecked())
        self.settings.setValue("upscale/factor", window.upscale_factor_combo.currentText())
        self.settings.setValue("upscale/method", window.upscale_method_combo.currentText())
    
    def load_all(self, window) -> None:
        """Load all settings into the window.
        
        Args:
            window: MainWindow instance to populate
        """
        self.load_window_geometry(window)
        self.load_recent_folders()
        self.load_params(window)
    
    def save_all(self, window) -> None:
        """Save all settings from the window.
        
        Args:
            window: MainWindow instance to read from
        """
        self.save_window_geometry(window)
        self.save_recent_folders()
        self.save_params(window)
    
    def save_single(self, key: str, value: Any) -> None:
        """Save a single setting immediately.
        
        Args:
            key: Setting key
            value: Value to save
        """
        self.settings.setValue(key, value)
        self.settings.sync()  # Ensure written to disk
    
    def reset_to_defaults(self, window) -> None:
        """Reset all parameters to default values.
        
        Args:
            window: MainWindow instance to reset
        """
        # Clear all params and OCR settings
        for key in DEFAULT_SETTINGS:
            self.settings.remove(key)
        
        self.settings.sync()
        
        # Reload into window
        self.load_params(window)
        
        logger.info("Settings reset to defaults")
    
    def get_default(self, key: str) -> Any:
        """Get default value for a setting.
        
        Args:
            key: Setting key
            
        Returns:
            Default value or None if not defined
        """
        return DEFAULT_SETTINGS.get(key)
