"""
Manhua Image Cleaner - Modern PyQt6 GUI
A tool for removing text from manhua/manga images using AI.
"""

import os
import sys

# Set KMP_DUPLICATE_LIB_OK to avoid OpenMP library conflicts on some systems
# This is a common issue when PyTorch is used with other libraries
if not os.environ.get("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QTextEdit, QProgressBar, QFileDialog, QTabWidget, QFrame,
    QScrollArea, QSplitter, QMessageBox, QGroupBox, QGridLayout,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QKeySequence, QAction, QShortcut, QDragEnterEvent, QDropEvent

# Module logger
logger = logging.getLogger(__name__)
from PIL import Image
import cv2
import numpy as np

from ..config import (
    IMAGE_CONFIG,
    SUPPORTED_IMAGE_EXTENSIONS,
    ModelType,
    Backend,
    WHITELIST_PRESETS,
)
from ..core import (
    BatchProcessor,
    ProcessingConfig,
)
from ..exceptions import (
    ValidationError,
    ImageProcessingError,
    OCRError,
    ModelError,
)
from ..utils.env import load_hf_token, save_hf_token, validate_hf_token

# Import GUI components
from .settings_manager import SettingsManager
from .preview_manager import PreviewManager, PREVIEW_MAX_DIMENSION
from .processing_controller import ProcessingController, ProcessingThread


# ============================================================================
# Color Scheme
# ============================================================================
class Theme:
    """Modern dark theme color scheme."""
    BG_DARK = "#0e1116"
    BG_MEDIUM = "#151a23"
    BG_LIGHT = "#1c2432"
    SURFACE = "#141a25"
    ACCENT = "#ff6a3d"
    ACCENT_HOVER = "#ff8a5b"
    ACCENT_SOFT = "#2a3646"
    HIGHLIGHT = "#5ed6c6"
    TEXT = "#e7ebf2"
    TEXT_DIM = "#a4adbb"
    SUCCESS = "#2ed573"
    WARNING = "#f1c40f"
    ERROR = "#ff4d4d"
    BORDER = "#263043"
    BORDER_LIGHT = "#33415c"


# ============================================================================
# Stylesheet
# ============================================================================
STYLESHEET = f"""
QMainWindow {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {Theme.BG_DARK}, stop:1 #121724);
    color: {Theme.TEXT};
    font-family: 'Space Grotesk', 'Manrope', 'Montserrat', sans-serif;
    font-size: 10pt;
}}

#centralWidget {{
    background: transparent;
}}

QLabel {{
    color: {Theme.TEXT};
}}

QGroupBox {{
    font-weight: 600;
    font-size: 10.5pt;
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 10px;
    background-color: {Theme.SURFACE};
    margin-top: 14px;
    padding: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: {Theme.TEXT_DIM};
    letter-spacing: 0.3px;
}}

QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {Theme.BG_LIGHT}, stop:1 {Theme.SURFACE});
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
}}

QPushButton:hover {{
    border: 1px solid {Theme.BORDER_LIGHT};
    background-color: {Theme.ACCENT_SOFT};
}}

QPushButton:pressed {{
    background-color: {Theme.BG_LIGHT};
}}

QPushButton:disabled {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT_DIM};
    border: 1px solid {Theme.BORDER};
}}

QPushButton#primaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Theme.ACCENT}, stop:1 {Theme.ACCENT_HOVER});
    color: #0d0f14;
    border: none;
    font-size: 11pt;
    padding: 12px 22px;
}}

QPushButton#primaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Theme.ACCENT_HOVER}, stop:1 {Theme.ACCENT});
}}

QPushButton#stopButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Theme.ERROR}, stop:1 #ff7a7a);
    color: #0d0f14;
    border: none;
}}

QLineEdit {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 8px 12px;
}}

QLineEdit:focus {{
    border: 1px solid {Theme.ACCENT};
}}

QComboBox {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    min-width: 160px;
}}

QComboBox:hover {{
    border: 1px solid {Theme.BORDER_LIGHT};
}}

QComboBox QAbstractItemView {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT};
    selection-background-color: {Theme.ACCENT};
    selection-color: #0d0f14;
    border: 1px solid {Theme.BORDER};
    outline: none;
}}

QSpinBox {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 8px 12px;
}}

QSpinBox:focus {{
    border: 1px solid {Theme.ACCENT};
}}

QCheckBox {{
    color: {Theme.TEXT};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1px solid {Theme.BORDER_LIGHT};
    background-color: {Theme.BG_MEDIUM};
}}

QCheckBox::indicator:checked {{
    background-color: {Theme.HIGHLIGHT};
    border-color: {Theme.HIGHLIGHT};
}}

QTextEdit {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 10px;
    padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 9pt;
}}

QProgressBar {{
    background-color: {Theme.BG_MEDIUM};
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    height: 10px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Theme.HIGHLIGHT}, stop:1 {Theme.ACCENT});
    border-radius: 6px;
}}

QTabWidget::pane {{
    border: 1px solid {Theme.BORDER};
    border-radius: 12px;
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: {Theme.BG_MEDIUM};
    color: {Theme.TEXT_DIM};
    padding: 10px 18px;
    margin-right: 4px;
    border-radius: 10px;
}}

QTabBar::tab:selected {{
    background-color: {Theme.SURFACE};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER_LIGHT};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {Theme.BG_MEDIUM};
    width: 10px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {Theme.BG_LIGHT};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {Theme.ACCENT};
}}

QSplitter::handle {{
    background-color: {Theme.BORDER};
}}

#previewLabel {{
    background-color: {Theme.BG_MEDIUM};
    border: 1px solid {Theme.BORDER};
    border-radius: 12px;
}}

#headerFrame {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #151b27, stop:1 #1b2230);
}}

#footerFrame {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #151b27, stop:1 #1b2230);
}}

#headerLabel {{
    font-size: 18pt;
    font-weight: 700;
    color: {Theme.ACCENT};
}}

#versionLabel {{
    font-size: 10pt;
    color: {Theme.TEXT_DIM};
}}

#statusLabel {{
    color: {Theme.TEXT_DIM};
}}
"""


# ============================================================================
# Main Window
# ============================================================================
class MainWindow(QMainWindow):
    """Modern GUI for the Manhua Image Cleaner application.
    
    This class uses component-based architecture:
    - SettingsManager: Handles settings persistence
    - PreviewManager: Manages preview display
    - ProcessingController: Manages processing lifecycle
    """
    
    ENV_FILE = ".env"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manhua Image Cleaner")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Accept drag and drop
        self.setAcceptDrops(True)
        
        # Initialize components
        self.settings_manager = SettingsManager(self)
        self.processing_controller = ProcessingController()
        
        # UI references (initialized in _create_ui)
        self.preview_manager: PreviewManager | None = None
        
        # Create UI
        self._create_ui()
        
        # Setup keyboard shortcuts
        self._setup_shortcuts()
        
        # Load saved settings
        self._load_settings()
        
        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)
        
        # Initial log
        self.log("Application started. Press Ctrl+Enter to start processing.", "info")
        self.log("Drag and drop images or folders here.", "info")
    
    def _load_hf_token(self) -> str:
        """Load HF token from .env file if it exists."""
        token = load_hf_token()
        return token or ""
    
    def _save_hf_token(self):
        """Save HF token to .env file."""
        try:
            save_hf_token(self.token_input.text())
            self.log("HuggingFace token saved successfully!", "success")
        except ValueError as e:
            self.log(f"Failed to save token: {e}", "error")
    
    def _create_ui(self):
        """Create the main user interface."""
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        self._create_header(main_layout)
        
        # Main content
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 10, 15, 10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Preview and Log
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (40% left, 60% right)
        splitter.setSizes([500, 700])
        
        content_layout.addWidget(splitter)
        main_layout.addWidget(content_widget, 1)
        
        # Footer
        self._create_footer(main_layout)
    
    def _create_header(self, parent_layout):
        """Create the header section."""
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setFixedHeight(60)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        title = QLabel("Manhua Image Cleaner")
        title.setObjectName("headerLabel")
        header_layout.addWidget(title)
        
        version = QLabel("v2.0 PyQt")
        version.setObjectName("versionLabel")
        header_layout.addWidget(version)
        
        header_layout.addStretch()
        
        parent_layout.addWidget(header)
    
    def _create_left_panel(self):
        """Create the left control panel with tabs."""
        tab_widget = QTabWidget()
        
        # Processing tab
        processing_tab = self._create_processing_tab()
        tab_widget.addTab(processing_tab, "  Processing  ")
        
        # Settings tab
        settings_tab = self._create_settings_tab()
        tab_widget.addTab(settings_tab, "  Settings  ")
        
        return tab_widget
    
    def _create_processing_tab(self):
        """Create the processing tab content."""
        from PyQt6.QtWidgets import QScrollArea
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Input Section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        self.input_path.setPlaceholderText("Select a file or folder...")
        self.input_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        input_layout.addWidget(self.input_path)
        
        btn_layout = QHBoxLayout()
        file_btn = QPushButton("Select File")
        file_btn.clicked.connect(self.select_file)
        btn_layout.addWidget(file_btn)
        
        folder_btn = QPushButton("Select Folder")
        folder_btn.clicked.connect(self.select_folder)
        btn_layout.addWidget(folder_btn)
        btn_layout.addStretch()
        input_layout.addLayout(btn_layout)
        
        layout.addWidget(input_group)
        
        # Output Section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        self.output_path.setPlaceholderText("Select output folder...")
        self.output_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_layout.addWidget(self.output_path)
        
        output_btn = QPushButton("Select Folder")
        output_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(output_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        layout.addWidget(output_group)
        
        # Parameters Section
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        params_layout.setSpacing(10)
        params_layout.setColumnStretch(1, 1)
        
        def set_expanding_policy(widget):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return widget
        
        # Model
        params_layout.addWidget(QLabel("Image Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([m.value for m in ModelType])
        set_expanding_policy(self.model_combo)
        params_layout.addWidget(self.model_combo, 0, 1)
        
        # Backend
        params_layout.addWidget(QLabel("Backend:"), 1, 0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([b.value for b in Backend])
        set_expanding_policy(self.backend_combo)
        params_layout.addWidget(self.backend_combo, 1, 1)
        
        # Steps
        params_layout.addWidget(QLabel("Steps:"), 2, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(4)
        set_expanding_policy(self.steps_spin)
        params_layout.addWidget(self.steps_spin, 2, 1)
        
        # Expand Pixels
        params_layout.addWidget(QLabel("AI Expand (px):"), 3, 0)
        self.expand_spin = QSpinBox()
        self.expand_spin.setRange(0, 500)
        self.expand_spin.setValue(IMAGE_CONFIG.default_expand_pixels)
        set_expanding_policy(self.expand_spin)
        params_layout.addWidget(self.expand_spin, 3, 1)
        
        # Smart Fill Expand
        params_layout.addWidget(QLabel("Smart Fill Expand (px):"), 4, 0)
        self.smart_fill_expand_spin = QSpinBox()
        self.smart_fill_expand_spin.setRange(0, 100)
        self.smart_fill_expand_spin.setValue(5)
        self.smart_fill_expand_spin.setToolTip("Smaller expansion for smart fill")
        set_expanding_policy(self.smart_fill_expand_spin)
        params_layout.addWidget(self.smart_fill_expand_spin, 4, 1)
        
        # Checkboxes
        check_layout = QGridLayout()
        check_layout.setHorizontalSpacing(12)
        check_layout.setVerticalSpacing(8)
        
        self.color_correct_check = QCheckBox("Color Correction")
        self.color_correct_check.setChecked(True)
        check_layout.addWidget(self.color_correct_check, 0, 0)
        
        self.edge_blend_check = QCheckBox("Edge Blending")
        self.edge_blend_check.setChecked(True)
        check_layout.addWidget(self.edge_blend_check, 0, 1)
        
        self.smart_fill_check = QCheckBox("Smart Fill")
        self.smart_fill_check.setChecked(True)
        check_layout.addWidget(self.smart_fill_check, 1, 0)
        
        self.debug_check = QCheckBox("Debug Mode")
        check_layout.addWidget(self.debug_check, 1, 1)
        check_layout.setColumnStretch(4, 1)
        
        params_layout.addLayout(check_layout, 5, 0, 1, 2)
        
        layout.addWidget(params_group)
        layout.addStretch()
        
        scroll.setWidget(content)
        return scroll
    
    def _create_settings_tab(self):
        """Create the settings tab content."""
        from PyQt6.QtWidgets import QScrollArea
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 15, 10, 10)
        
        def set_expanding_policy(widget):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return widget
        
        # HuggingFace Token Section
        token_group = QGroupBox("HuggingFace Token")
        token_layout = QVBoxLayout(token_group)
        
        desc = QLabel("Enter your HuggingFace API token to access the image models.\n"
                      "Get your token from: https://huggingface.co/settings/tokens")
        desc.setWordWrap(True)
        token_layout.addWidget(desc)
        
        token_input_layout = QHBoxLayout()
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setText(self._load_hf_token())
        self.token_input.setPlaceholderText("Enter your HF token...")
        set_expanding_policy(self.token_input)
        token_input_layout.addWidget(self.token_input)
        
        self.show_token_check = QCheckBox("Show")
        self.show_token_check.toggled.connect(self._toggle_token_visibility)
        token_input_layout.addWidget(self.show_token_check)
        token_layout.addLayout(token_input_layout)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Token")
        save_btn.clicked.connect(self._save_hf_token)
        btn_layout.addWidget(save_btn)
        
        test_btn = QPushButton("Test Token")
        test_btn.clicked.connect(self._test_hf_token)
        btn_layout.addWidget(test_btn)
        btn_layout.addStretch()
        token_layout.addLayout(btn_layout)
        
        layout.addWidget(token_group)
        
        # OCR Settings Section
        ocr_group = QGroupBox("OCR Settings")
        ocr_layout = QVBoxLayout(ocr_group)
        
        ocr_desc = QLabel("Configure text detection (OCR) settings.")
        ocr_desc.setWordWrap(True)
        ocr_layout.addWidget(ocr_desc)
        
        # OCR Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("OCR Model:"))
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.addItems(["paddleocr", "easyocr"])
        self.ocr_model_combo.setCurrentText("paddleocr")
        self.ocr_model_combo.setToolTip(
            "paddleocr: Best for Asian text, requires PaddlePaddle (default)\n"
            "easyocr: Supports 80+ languages, easier to install"
        )
        set_expanding_policy(self.ocr_model_combo)
        model_layout.addWidget(self.ocr_model_combo, 1)
        ocr_layout.addLayout(model_layout)
        
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("Pipeline Version:"))
        self.ocr_version_combo = QComboBox()
        self.ocr_version_combo.addItems(["v1.5", "v1.0"])
        self.ocr_version_combo.setCurrentText("v1.5")
        set_expanding_policy(self.ocr_version_combo)
        version_layout.addWidget(self.ocr_version_combo, 1)
        ocr_layout.addLayout(version_layout)
        
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Precision:"))
        self.ocr_precision_combo = QComboBox()
        self.ocr_precision_combo.addItems(["fp16", "fp32"])
        self.ocr_precision_combo.setCurrentText("fp16")
        set_expanding_policy(self.ocr_precision_combo)
        precision_layout.addWidget(self.ocr_precision_combo, 1)
        ocr_layout.addLayout(precision_layout)
        
        self.ocr_tensorrt_check = QCheckBox("Use TensorRT (if available)")
        self.ocr_tensorrt_check.setChecked(True)
        ocr_layout.addWidget(self.ocr_tensorrt_check)
        
        # OCR Workers - with warning about VRAM
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("OCR Workers:"))
        self.ocr_workers_spin = QSpinBox()
        self.ocr_workers_spin.setRange(1, 8)
        self.ocr_workers_spin.setValue(1)
        self.ocr_workers_spin.setSuffix(" workers")
        self.ocr_workers_spin.setToolTip(
            "Number of parallel OCR worker processes.\n"
            "Each worker loads a full OCR model (~16GB VRAM).\n"
            "Increase only if you have sufficient VRAM."
        )
        set_expanding_policy(self.ocr_workers_spin)
        workers_layout.addWidget(self.ocr_workers_spin, 1)
        ocr_layout.addLayout(workers_layout)
        
        # VRAM warning label
        vram_warning = QLabel("⚠️ Each OCR worker uses ~16GB VRAM. Use 1 for most systems.")
        vram_warning.setStyleSheet("color: #f1c40f; font-size: 9pt;")
        ocr_layout.addWidget(vram_warning)
        
        layout.addWidget(ocr_group)
        
        # Whitelist Settings Section
        whitelist_group = QGroupBox("Character Whitelist")
        whitelist_layout = QVBoxLayout(whitelist_group)
        
        whitelist_desc = QLabel(
            "Preserve text matching certain patterns. "
            "Nearby text regions are grouped and checked together."
        )
        whitelist_desc.setWordWrap(True)
        whitelist_layout.addWidget(whitelist_desc)
        
        # Enable checkbox
        self.whitelist_enable_check = QCheckBox("Enable Whitelist")
        self.whitelist_enable_check.setChecked(False)
        whitelist_layout.addWidget(self.whitelist_enable_check)
        
        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.whitelist_preset_combo = QComboBox()
        self.whitelist_preset_combo.addItems(list(WHITELIST_PRESETS.keys()))
        self.whitelist_preset_combo.setCurrentText("none")
        self.whitelist_preset_combo.setToolTip(
            "Select a preset pattern set:\n"
            "- none: No whitelisting\n"
            "- sfx_only: SFX symbols (!?…〜)\n"
            "- punctuation: Ending punctuation\n"
            "- hearts: Heart symbols (♡♥❤)\n"
            "- symbols: Music notes, stars, etc.\n"
            "- japanese_sfx: Japanese SFX characters"
        )
        set_expanding_policy(self.whitelist_preset_combo)
        preset_layout.addWidget(self.whitelist_preset_combo, 1)
        whitelist_layout.addLayout(preset_layout)
        
        # Group distance
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Group Distance:"))
        self.whitelist_distance_spin = QSpinBox()
        self.whitelist_distance_spin.setRange(10, 200)
        self.whitelist_distance_spin.setValue(50)
        self.whitelist_distance_spin.setSuffix(" px")
        self.whitelist_distance_spin.setToolTip(
            "Maximum distance between text regions to be grouped as one textbox.\n"
            "Larger values group more regions together."
        )
        set_expanding_policy(self.whitelist_distance_spin)
        distance_layout.addWidget(self.whitelist_distance_spin, 1)
        whitelist_layout.addLayout(distance_layout)
        
        # Custom characters/words
        whitelist_layout.addWidget(QLabel("Custom Characters/Words (one per line):"))
        self.whitelist_custom_input = QTextEdit()
        self.whitelist_custom_input.setPlaceholderText(
            "Enter characters or words to preserve, one per line:\n"
            "!?♡♥…\n"
            "SFX\n"
            "!!!"
        )
        self.whitelist_custom_input.setMaximumHeight(100)
        whitelist_layout.addWidget(self.whitelist_custom_input)
        
        # Help text
        help_label = QLabel(
            "Tip: Each line becomes a whitelist pattern. "
            "If a textbox contains ONLY these characters, it won't be cleaned."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #a4adbb; font-size: 9pt;")
        whitelist_layout.addWidget(help_label)
        
        layout.addWidget(whitelist_group)
        
        # Prompts Section
        prompts_group = QGroupBox("Prompts")
        prompts_layout = QVBoxLayout(prompts_group)
        
        prompts_layout.addWidget(QLabel("Main Prompt:"))
        self.prompt_input = QLineEdit("remove all text")
        set_expanding_policy(self.prompt_input)
        prompts_layout.addWidget(self.prompt_input)
        
        self.extra_pass_check = QCheckBox("Enable Extra Pass")
        prompts_layout.addWidget(self.extra_pass_check)
        
        prompts_layout.addWidget(QLabel("Extra Pass Prompt:"))
        self.extra_prompt_input = QLineEdit("improve image quality")
        set_expanding_policy(self.extra_prompt_input)
        prompts_layout.addWidget(self.extra_prompt_input)
        
        # Extra Pass Upscaling
        upscale_group = QGroupBox("Extra Pass Upscaling")
        upscale_layout = QVBoxLayout(upscale_group)
        
        self.extra_pass_upscale_check = QCheckBox("Upscale before extra pass")
        self.extra_pass_upscale_check.setToolTip(
            "Resize image before the extra quality pass. "
            "The AI model will add detail when processing the larger image."
        )
        upscale_layout.addWidget(self.extra_pass_upscale_check)
        
        # Upscale factor
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel("Scale:"))
        self.upscale_factor_combo = QComboBox()
        self.upscale_factor_combo.addItems(["1.5x", "2x", "3x", "4x", "8x"])
        self.upscale_factor_combo.setCurrentText("2x")
        self.upscale_factor_combo.setToolTip("How much to upscale the image")
        set_expanding_policy(self.upscale_factor_combo)
        factor_layout.addWidget(self.upscale_factor_combo)
        upscale_layout.addLayout(factor_layout)
        
        # Upscale method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.upscale_method_combo = QComboBox()
        self.upscale_method_combo.addItems(["lanczos", "bicubic", "bilinear"])
        self.upscale_method_combo.setCurrentText("lanczos")
        self.upscale_method_combo.setToolTip(
            "Lanczos = best quality, slower\n"
            "Bicubic = good balance\n"
            "Bilinear = fastest"
        )
        set_expanding_policy(self.upscale_method_combo)
        method_layout.addWidget(self.upscale_method_combo)
        upscale_layout.addLayout(method_layout)
        
        prompts_layout.addWidget(upscale_group)
        
        layout.addWidget(prompts_group)
        
        # About Section
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        
        about_text = """Manhua Image Cleaner v2.0 (PyQt Edition)

AI-powered text removal from manga, manhua, and comic images.

Features:
- OCR-based text detection (PaddleOCR)
- AI-powered inpainting (FLUX.2 / LongCat)
- Batch processing support
- Color correction and edge blending
- Smart fill for simple backgrounds"""
        
        about_label = QLabel(about_text)
        about_label.setWordWrap(True)
        about_layout.addWidget(about_label)
        
        layout.addWidget(about_group)
        
        # Reset Settings Section
        reset_group = QGroupBox("Reset Settings")
        reset_layout = QVBoxLayout(reset_group)
        
        reset_desc = QLabel("Reset all settings to their default values.\n"
                           "This will not affect your HuggingFace token.")
        reset_desc.setWordWrap(True)
        reset_layout.addWidget(reset_desc)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_settings_to_defaults)
        reset_layout.addWidget(reset_btn)
        
        layout.addWidget(reset_group)
        layout.addStretch()
        
        scroll.setWidget(content)
        return scroll
    
    def _toggle_token_visibility(self, checked):
        """Toggle token input visibility."""
        if checked:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
    
    def _test_hf_token(self):
        """Test if the HF token is valid."""
        token = self.token_input.text().strip()
        if not token:
            self.log("Please enter a token first", "warning")
            return
        
        self.log("Testing HuggingFace token...", "info")
        
        import threading
        def test():
            result = validate_hf_token(token)
            if result["valid"]:
                self.log(f"Token valid! Logged in as: {result['username']}", "success")
            else:
                self.log(f"Invalid token: {result['error']}", "error")
        
        threading.Thread(target=test, daemon=True).start()
    
    def _create_right_panel(self):
        """Create the right panel with preview and log."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Image Preview Section
        preview_group = QGroupBox("Image Preview (Previous vs Current)")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Previous
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.prev_stage_label = QLabel("Previous: -")
        self.prev_stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_stage_label.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-weight: bold; font-size: 10pt; padding: 3px;"
        )
        left_layout.addWidget(self.prev_stage_label)
        
        self.prev_preview_scroll = QScrollArea()
        self.prev_preview_scroll.setWidgetResizable(True)
        self.prev_preview_scroll.setMinimumWidth(350)
        self.prev_preview_scroll.setMinimumHeight(500)
        
        self.prev_preview_label = QLabel("No image")
        self.prev_preview_label.setObjectName("previewLabel")
        self.prev_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_preview_label.setStyleSheet(
            f"background-color: {Theme.BG_MEDIUM}; border-radius: 12px; padding: 6px;"
        )
        self.prev_preview_scroll.setWidget(self.prev_preview_label)
        left_layout.addWidget(self.prev_preview_scroll)
        
        self.preview_splitter.addWidget(left_widget)
        
        # Right side - Current
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.curr_stage_label = QLabel("Current: -")
        self.curr_stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.curr_stage_label.setStyleSheet(
            f"color: {Theme.HIGHLIGHT}; font-weight: bold; font-size: 10pt; padding: 3px;"
        )
        right_layout.addWidget(self.curr_stage_label)
        
        self.curr_preview_scroll = QScrollArea()
        self.curr_preview_scroll.setWidgetResizable(True)
        self.curr_preview_scroll.setMinimumWidth(350)
        self.curr_preview_scroll.setMinimumHeight(500)
        
        self.curr_preview_label = QLabel("No image loaded")
        self.curr_preview_label.setObjectName("previewLabel")
        self.curr_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.curr_preview_label.setStyleSheet(
            f"background-color: {Theme.BG_MEDIUM}; border-radius: 12px; padding: 6px;"
        )
        self.curr_preview_scroll.setWidget(self.curr_preview_label)
        right_layout.addWidget(self.curr_preview_scroll)
        
        self.preview_splitter.addWidget(right_widget)
        
        # Set equal sizes
        self.preview_splitter.setSizes([400, 400])
        
        preview_layout.addWidget(self.preview_splitter)
        layout.addWidget(preview_group, 3)
        
        # Create preview manager
        self.preview_manager = PreviewManager(
            prev_scroll=self.prev_preview_scroll,
            prev_label=self.prev_preview_label,
            prev_stage_label=self.prev_stage_label,
            curr_scroll=self.curr_preview_scroll,
            curr_label=self.curr_preview_label,
            curr_stage_label=self.curr_stage_label
        )
        
        # Activity Log Section
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        # Log toolbar
        log_toolbar = QHBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setToolTip("Clear log (Ctrl+L)")
        clear_log_btn.clicked.connect(self._clear_log)
        log_toolbar.addWidget(clear_log_btn)
        
        copy_log_btn = QPushButton("Copy")
        copy_log_btn.setToolTip("Copy log to clipboard")
        copy_log_btn.clicked.connect(self._copy_log)
        log_toolbar.addWidget(copy_log_btn)
        
        shortcuts_btn = QPushButton("Shortcuts")
        shortcuts_btn.setToolTip("Show keyboard shortcuts")
        shortcuts_btn.clicked.connect(self._show_shortcuts)
        log_toolbar.addWidget(shortcuts_btn)
        
        help_btn = QPushButton("Help")
        help_btn.setToolTip("Show quick start guide")
        help_btn.clicked.connect(self._show_help)
        log_toolbar.addWidget(help_btn)
        
        log_toolbar.addStretch()
        
        # Image counter label
        self.image_counter_label = QLabel("")
        self.image_counter_label.setStyleSheet(f"color: {Theme.TEXT_DIM};")
        log_toolbar.addWidget(self.image_counter_label)
        
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group, 1)
        
        return panel
    
    def _create_footer(self, parent_layout):
        """Create the footer with progress and buttons."""
        footer = QFrame()
        footer.setObjectName("footerFrame")
        footer.setFixedHeight(100)
        
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(20, 15, 20, 15)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        footer_layout.addWidget(self.progress_bar)
        
        # Status and buttons
        bottom_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        bottom_layout.addWidget(self.status_label)
        
        bottom_layout.addStretch()
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        bottom_layout.addWidget(self.stop_btn)
        
        self.process_btn = QPushButton("Process Images")
        self.process_btn.setObjectName("primaryButton")
        self.process_btn.clicked.connect(self.process_images)
        bottom_layout.addWidget(self.process_btn)
        
        footer_layout.addLayout(bottom_layout)
        
        parent_layout.addWidget(footer)
    
    def log(self, message: str, level: str = "info"):
        """Add a message to the log with timestamp and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": Theme.TEXT,
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "debug": Theme.TEXT_DIM
        }
        color = colors.get(level, Theme.TEXT)
        
        html = f'<span style="color: {Theme.TEXT_DIM}">[{timestamp}]</span> <span style="color: {color}">{message}</span><br>'
        self.log_text.insertHtml(html)
        self.log_text.ensureCursorVisible()
    
    def _update_file_counter(self, current: int, total: int, filename: str):
        """Update the file counter display."""
        self.image_counter_label.setText(f"Image {current} of {total}: {filename}")
    
    def _update_stats(self, smart_filled: int, ai_processed: int):
        """Update running statistics."""
        self.processing_controller.update_stats(smart_filled, ai_processed)
    
    def _on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        
        # Show final stats
        total_smart, total_ai = self.processing_controller.get_stats()
        total_boxes = total_smart + total_ai
        if total_boxes > 0:
            self.log("=" * 50, "info")
            self.log(
                f"Total: {total_smart} smart filled, "
                f"{total_ai} AI processed "
                f"({100*total_smart/total_boxes:.0f}% optimized)",
                "success" if success else "warning"
            )
        
        if success:
            self.status_label.setText("Complete!")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Finished with errors")
        
        # Cleanup
        self.processing_controller.cleanup()
    
    def select_file(self):
        """Open file dialog to select an image file."""
        start_dir = self.settings_manager.get_recent_input_folder()
        
        extensions = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            start_dir,
            f"Image Files ({extensions})"
        )
        if file_path:
            self.input_path.setText(file_path)
            self.settings_manager.add_recent_input_folder(str(Path(file_path).parent))
            self.log(f"Selected file: {Path(file_path).name}", "info")
            
            # Load preview
            img = cv2.imread(file_path)
            if img is not None:
                self.preview_manager.set_original_preview(img)
    
    def select_folder(self):
        """Open folder dialog to select input folder."""
        start_dir = self.settings_manager.get_recent_input_folder()
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Input Folder", start_dir)
        if folder_path:
            self.input_path.setText(folder_path)
            self.settings_manager.add_recent_input_folder(folder_path)
            count = sum(1 for f in Path(folder_path).iterdir() 
                       if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
            self.log(f"Selected folder: {folder_path} ({count} images)", "info")
    
    def select_output_folder(self):
        """Open folder dialog to select output folder."""
        start_dir = self.settings_manager.get_recent_output_folder()
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if folder_path:
            self.output_path.setText(folder_path)
            self.settings_manager.add_recent_output_folder(folder_path)
            self.log(f"Output folder: {folder_path}", "info")
    
    def _validate_inputs(self) -> bool:
        """Validate all inputs before processing."""
        if not self.input_path.text():
            QMessageBox.warning(self, "Validation Error", "Please select an input file or folder")
            return False
        
        if not self.output_path.text():
            QMessageBox.warning(self, "Validation Error", "Please select an output folder")
            return False
        
        token = self.token_input.text().strip()
        if not token:
            result = QMessageBox.question(
                self,
                "Missing Token",
                "No HuggingFace token configured.\n\nImage models require a valid token. Continue anyway?"
            )
            if result != QMessageBox.StandardButton.Yes:
                return False
            self.log("Proceeding without HF token", "warning")
        else:
            import os
            os.environ['HF_TOKEN'] = token
        
        return True
    
    def _get_files(self) -> list[Path]:
        """Get list of files to process."""
        input_path = Path(self.input_path.text())
        
        if input_path.is_file():
            return [input_path]
        
        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return sorted(files)
    
    def process_images(self):
        """Start image processing."""
        if not self._validate_inputs():
            return
        
        files = self._get_files()
        if not files:
            QMessageBox.warning(self, "Warning", "No image files found")
            return
        
        # Create config
        extra_prompt = None
        if self.extra_pass_check.isChecked():
            extra_prompt = self.extra_prompt_input.text()
        
        # Convert custom whitelist text to regex patterns
        import re
        custom_lines = [
            line.strip() for line in self.whitelist_custom_input.toPlainText().split('\n')
            if line.strip()
        ]
        # Convert each line to a regex pattern that matches text containing only those chars
        # Escape special regex characters and create a pattern like: ^[\schars]*$ 
        # The \s* allows optional whitespace around characters
        custom_patterns = []
        for line in custom_lines:
            # Escape special regex characters within the character set
            escaped = ''.join(re.escape(c) for c in line)
            # Pattern allows: only these chars, with optional whitespace mixed in
            pattern = f'^[\\s{escaped}]*$'
            custom_patterns.append(pattern)
        
        config = ProcessingConfig(
            model_type=ModelType(self.model_combo.currentText()),
            device=self.backend_combo.currentText(),
            steps=self.steps_spin.value(),
            expand_pixels=self.expand_spin.value(),
            color_correct=self.color_correct_check.isChecked(),
            edge_blend=self.edge_blend_check.isChecked(),
            smart_fill=self.smart_fill_check.isChecked(),
            smart_fill_expand_pixels=self.smart_fill_expand_spin.value(),
            ocr_model=self.ocr_model_combo.currentText(),
            ocr_pipeline_version=self.ocr_version_combo.currentText(),
            ocr_use_tensorrt=self.ocr_tensorrt_check.isChecked(),
            ocr_precision=self.ocr_precision_combo.currentText(),
            ocr_workers=self.ocr_workers_spin.value(),
            prompt=self.prompt_input.text(),
            extra_pass_prompt=extra_prompt,
            whitelist_enabled=self.whitelist_enable_check.isChecked(),
            whitelist_preset=self.whitelist_preset_combo.currentText(),
            whitelist_patterns=custom_patterns,
            whitelist_group_distance=self.whitelist_distance_spin.value(),
            extra_pass_upscale=self.extra_pass_upscale_check.isChecked(),
            extra_pass_upscale_factor=float(self.upscale_factor_combo.currentText().replace('x', '')),
            extra_pass_upscale_method=self.upscale_method_combo.currentText()
        )
        
        # Log configuration
        self.log("Configuration:", "info")
        self.log(f"  Model: {config.model_type.value}", "info")
        self.log(f"  Device: {config.device}", "info")
        self.log(f"  Steps: {config.steps}", "info")
        self.log(f"  Smart Fill: {config.smart_fill}", "info")
        
        # Update UI state
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.log("=" * 50, "info")
        self.log(f"Starting batch processing: {len(files)} images", "info")
        
        # Start processing
        thread = self.processing_controller.start_processing(
            config=config,
            files=files,
            output_dir=Path(self.output_path.text()),
            debug_mode=self.debug_check.isChecked()
        )
        
        # Connect signals
        thread.log_signal.connect(self.log)
        thread.status_signal.connect(self.status_label.setText)
        thread.progress_signal.connect(self._update_progress)
        thread.file_progress_signal.connect(self._update_file_counter)
        thread.preview_signal.connect(self.preview_manager.queue_update)
        thread.stats_signal.connect(self._update_stats)
        thread.finished_signal.connect(self._on_processing_finished)
        thread.start()
    
    def _update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
    
    def stop_processing(self):
        """Stop the current processing job."""
        self.log("Stop requested...", "warning")
        self.processing_controller.stop_processing()
        self.status_label.setText("Stopping...")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings before closing
        self._save_settings()
        
        # Cleanup processing
        self.processing_controller.cleanup()
        
        event.accept()
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Ctrl+Enter - Start processing
        self.process_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.process_shortcut.activated.connect(self._on_process_shortcut)
        
        # Escape - Stop processing
        self.stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_shortcut.activated.connect(self._on_stop_shortcut)
        
        # Ctrl+L - Clear log
        self.clear_log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_log_shortcut.activated.connect(self._clear_log)
        
        # Ctrl+O - Open file
        self.open_file_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.open_file_shortcut.activated.connect(self.select_file)
        
        # Ctrl+D - Open folder
        self.open_folder_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self.open_folder_shortcut.activated.connect(self.select_folder)
    
    def _on_process_shortcut(self):
        """Handle Ctrl+Enter shortcut."""
        if self.process_btn.isEnabled():
            self.process_images()
    
    def _on_stop_shortcut(self):
        """Handle Escape shortcut."""
        if self.stop_btn.isEnabled():
            self.stop_processing()
    
    def _load_settings(self):
        """Load saved settings and connect auto-save signals."""
        self.settings_manager.load_all(self)
        self._connect_auto_save_signals()
    
    def _save_settings(self):
        """Save current settings."""
        self.settings_manager.save_all(self)
    
    def _connect_auto_save_signals(self):
        """Connect UI control signals to auto-save settings."""
        # Model params
        self.model_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("params/model", text)
        )
        self.backend_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("params/backend", text)
        )
        self.steps_spin.valueChanged.connect(
            lambda val: self.settings_manager.save_single("params/steps", val)
        )
        self.expand_spin.valueChanged.connect(
            lambda val: self.settings_manager.save_single("params/expand", val)
        )
        
        # Checkboxes
        self.smart_fill_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("params/smart_fill", checked)
        )
        self.smart_fill_expand_spin.valueChanged.connect(
            lambda val: self.settings_manager.save_single("params/smart_fill_expand", val)
        )
        self.color_correct_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("params/color_correct", checked)
        )
        self.edge_blend_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("params/edge_blend", checked)
        )
        
        # Prompt
        self.prompt_input.textChanged.connect(
            lambda text: self.settings_manager.save_single("params/prompt", text)
        )
        
        # OCR settings
        self.ocr_model_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("ocr/model", text)
        )
        self.ocr_workers_spin.valueChanged.connect(
            lambda val: self.settings_manager.save_single("ocr/workers", val)
        )
        self.ocr_version_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("ocr/version", text)
        )
        self.ocr_precision_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("ocr/precision", text)
        )
        self.ocr_tensorrt_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("ocr/tensorrt", checked)
        )
        
        # Whitelist settings
        self.whitelist_enable_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("whitelist/enabled", checked)
        )
        self.whitelist_preset_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("whitelist/preset", text)
        )
        self.whitelist_distance_spin.valueChanged.connect(
            lambda val: self.settings_manager.save_single("whitelist/distance", val)
        )
        self.whitelist_custom_input.textChanged.connect(
            lambda: self.settings_manager.save_single("whitelist/custom", self.whitelist_custom_input.toPlainText())
        )
        
        # Extra pass upscaling settings
        self.extra_pass_upscale_check.toggled.connect(
            lambda checked: self.settings_manager.save_single("upscale/enabled", checked)
        )
        self.upscale_factor_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("upscale/factor", text)
        )
        self.upscale_method_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.save_single("upscale/method", text)
        )
    
    def _reset_settings_to_defaults(self):
        """Reset all settings to default values."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?\n\n"
            "This will not affect your HuggingFace token.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings_manager.reset_to_defaults(self)
            self.log("Settings reset to default values.", "success")
    
    def _clear_log(self):
        """Clear the activity log."""
        self.log_text.clear()
        self.log("Log cleared.", "info")
    
    def _copy_log(self):
        """Copy log contents to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())
        self.log("Log copied to clipboard.", "success")
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td><b>Ctrl+Enter</b></td><td>Start processing</td></tr>
        <tr><td><b>Escape</b></td><td>Stop processing</td></tr>
        <tr><td><b>Ctrl+O</b></td><td>Open file</td></tr>
        <tr><td><b>Ctrl+D</b></td><td>Open folder</td></tr>
        <tr><td><b>Ctrl+L</b></td><td>Clear log</td></tr>
        </table>
        <br>
        <h3>Drag & Drop</h3>
        <p>You can drag and drop:</p>
        <ul>
        <li>Single image file</li>
        <li>Multiple image files</li>
        <li>Folders containing images</li>
        </ul>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(shortcuts_text)
        msg.exec()
    
    def _show_help(self):
        """Show quick start help dialog."""
        help_text = """
        <h2>🚀 Quick Start Guide</h2>
        
        <h3>Step 1: Get Your Free Token (One-time)</h3>
        <ol>
        <li>Visit: <a href='https://huggingface.co/settings/tokens'>huggingface.co/settings/tokens</a></li>
        <li>Click "New token" → Name it → Copy the token</li>
        <li>Go to <b>Settings</b> tab in this app</li>
        <li>Paste token → Click <b>"Save Token"</b></li>
        </ol>
        
        <h3>Step 2: Select Images</h3>
        <ul>
        <li><b>Main</b> tab → Click <b>"Select File"</b> or <b>"Select Folder"</b></li>
        <li>Or drag & drop images into the window</li>
        </ul>
        
        <h3>Step 3: Select Output Folder</h3>
        <ul>
        <li>Click <b>"Select Folder"</b> in Output section</li>
        <li>Cleaned images will be saved here</li>
        </ul>
        
        <h3>Step 4: Start Processing</h3>
        <ul>
        <li>Click <b>"Process Images"</b> or press <b>Ctrl+Enter</b></li>
        <li>Wait for processing to complete</li>
        </ul>
        
        <h3>Recommended Settings</h3>
        <table border='1' cellpadding='5'>
        <tr><td><b>Image Model</b></td><td>FLUX.2-klein-4B</td></tr>
        <tr><td><b>Steps</b></td><td>4</td></tr>
        <tr><td><b>AI Expand</b></td><td>25</td></tr>
        <tr><td><b>Smart Fill</b></td><td>✓ ON</td></tr>
        </table>
        
        <h3>What is the Whitelist?</h3>
        <p>Use it to <b>keep certain text</b> (like sound effects):</p>
        <ol>
        <li>Go to <b>Settings</b> tab</li>
        <li>Check <b>"Enable Whitelist"</b></li>
        <li>Select a preset (e.g., <b>sfx_only</b> keeps !? symbols)</li>
        <li>Add custom characters if needed</li>
        </ol>
        
        <h3>Common Issues</h3>
        <table border='1' cellpadding='5'>
        <tr><td><b>Too slow?</b></td><td>Reduce Steps to 4, use 4B model</td></tr>
        <tr><td><b>Bad quality?</b></td><td>Increase Steps to 8-12</td></tr>
        <tr><td><b>Crashes?</b></td><td>Set OCR Workers to 1</td></tr>
        <tr><td><b>Text not found?</b></td><td>Use PaddleOCR (not EasyOCR)</td></tr>
        </table>
        
        <p><i>See USER_GUIDE.md for detailed documentation.</i></p>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Quick Start Guide")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Handle drag move event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if not urls:
            return
        
        paths = [url.toLocalFile() for url in urls]
        
        # Filter for images or folders
        image_paths = []
        folders = []
        
        for path in paths:
            p = Path(path)
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(path)
            elif p.is_dir():
                folders.append(path)
        
        # Handle based on what was dropped
        if len(image_paths) == 1 and not folders:
            # Single image
            self.input_path.setText(image_paths[0])
            self.settings_manager.add_recent_input_folder(str(Path(image_paths[0]).parent))
            self.log(f"Dropped file: {Path(image_paths[0]).name}", "info")
            
            # Load preview
            img = cv2.imread(image_paths[0])
            if img is not None:
                self.preview_manager.set_original_preview(img)
                
        elif len(image_paths) > 1:
            # Multiple images - set parent folder as input
            parent = str(Path(image_paths[0]).parent)
            self.input_path.setText(parent)
            self.settings_manager.add_recent_input_folder(parent)
            self.log(f"Dropped {len(image_paths)} images", "info")
            
        elif folders:
            # Folder(s) dropped
            if len(folders) == 1:
                self.input_path.setText(folders[0])
                self.settings_manager.add_recent_input_folder(folders[0])
                count = sum(1 for f in Path(folders[0]).iterdir() 
                           if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
                self.log(f"Dropped folder: {folders[0]} ({count} images)", "info")
            else:
                self.log(f"Dropped {len(folders)} folders", "info")


def main():
    """Main entry point."""
    import multiprocessing
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
