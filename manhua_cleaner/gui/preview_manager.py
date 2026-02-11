"""Preview display manager for the GUI."""

import logging
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QScrollArea, QLabel

logger = logging.getLogger(__name__)

# Preview constants
PREVIEW_MARGIN = 30
PREVIEW_MIN_WIDTH = 300
PREVIEW_MIN_HEIGHT = 400
PREVIEW_MAX_DIMENSION = 1200


class PreviewManager:
    """Manages image preview display with throttling and memory safety.
    
    This class handles:
    - Throttled preview updates to prevent UI freezing
    - Side-by-side comparison display
    - Memory-safe QImage conversion
    - Automatic scaling to fit display area
    """
    
    def __init__(
        self,
        prev_scroll: QScrollArea,
        prev_label: QLabel,
        prev_stage_label: QLabel,
        curr_scroll: QScrollArea,
        curr_label: QLabel,
        curr_stage_label: QLabel,
        throttle_ms: int = 50
    ):
        """Initialize preview manager.
        
        Args:
            prev_scroll: Scroll area for previous/left preview
            prev_label: Label for previous/left preview image
            prev_stage_label: Label for previous/left stage text
            curr_scroll: Scroll area for current/right preview
            curr_label: Label for current/right preview image
            curr_stage_label: Label for current/right stage text
            throttle_ms: Throttle interval in milliseconds
        """
        self.prev_scroll = prev_scroll
        self.prev_label = prev_label
        self.prev_stage_label = prev_stage_label
        self.curr_scroll = curr_scroll
        self.curr_label = curr_label
        self.curr_stage_label = curr_stage_label
        
        # Throttling
        self._pending_preview: tuple[np.ndarray, str, bool] | None = None
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(throttle_ms)
        self._preview_timer.timeout.connect(self._process_pending_preview)
        
        # Stage tracking
        self._is_pre_ai = False
    
    def queue_update(
        self,
        img_array: np.ndarray,
        stage: str = "",
        is_pre_ai: bool = False
    ) -> None:
        """Queue a preview update (throttled).
        
        Args:
            img_array: Image data in BGR format
            stage: Processing stage label
            is_pre_ai: If True, this is the pre-AI consolidated view for left side
        """
        if img_array is None:
            return
        
        self._pending_preview = (img_array.copy(), stage, is_pre_ai)
        self._preview_timer.start()
    
    def _process_pending_preview(self) -> None:
        """Process the most recently queued preview update."""
        if self._pending_preview is None:
            return
        
        img_array, stage, is_pre_ai = self._pending_preview
        self._pending_preview = None
        
        try:
            if is_pre_ai:
                self._update_pre_ai_preview(img_array, stage)
            elif self._is_side_by_side_stage(stage):
                self._update_side_by_side_preview(img_array, stage)
            else:
                self._update_current_preview(img_array, stage)
        except Exception as e:
            logger.warning(f"Preview update error: {e}", exc_info=True)
    
    def _is_side_by_side_stage(self, stage: str) -> bool:
        """Check if the stage indicates a side-by-side comparison image.
        
        Args:
            stage: Processing stage label to check.
            
        Returns:
            True if this is a side-by-side comparison stage.
        """
        stage_lower = stage.lower()
        return "ai_region_" in stage_lower or "before" in stage_lower
    
    def _update_pre_ai_preview(self, img_array: np.ndarray, stage: str) -> None:
        """Update the pre-AI preview on the left side.
        
        Args:
            img_array: Image data in BGR format.
            stage: Processing stage label.
        """
        pixmap = self._create_scaled_pixmap(img_array, self.prev_scroll)
        self.prev_label.setPixmap(pixmap)
        self.prev_stage_label.setText(f"Previous: {stage}")
        self._is_pre_ai = True
    
    def _update_side_by_side_preview(self, img_array: np.ndarray, stage: str) -> None:
        """Update both preview sides from a side-by-side comparison image.
        
        Args:
            img_array: Side-by-side image data in BGR format.
            stage: Processing stage label.
        """
        half_width = img_array.shape[1] // 2
        left_half = img_array[:, :half_width]
        right_half = img_array[:, half_width:]
        
        # Update left side (before)
        left_pixmap = self._create_scaled_pixmap(left_half, self.prev_scroll)
        self.prev_label.setPixmap(left_pixmap)
        self.prev_stage_label.setText("Previous: Before AI")
        
        # Update right side (after)
        right_pixmap = self._create_scaled_pixmap(right_half, self.curr_scroll)
        self.curr_label.setPixmap(right_pixmap)
        self.curr_stage_label.setText(f"Current: {stage}")
    
    def _update_current_preview(self, img_array: np.ndarray, stage: str) -> None:
        """Update the current preview on the right side.
        
        Args:
            img_array: Image data in BGR format.
            stage: Processing stage label.
        """
        pixmap = self._create_scaled_pixmap(img_array, self.curr_scroll)
        self.curr_label.setPixmap(pixmap)
        self.curr_stage_label.setText(f"Current: {stage}")
    
    def _create_scaled_pixmap(
        self,
        img_array: np.ndarray,
        scroll_area: QScrollArea
    ) -> QPixmap:
        """Convert BGR image to RGB and create a scaled QPixmap.
        
        Args:
            img_array: BGR image as numpy array
            scroll_area: QScrollArea to scale for
            
        Returns:
            Scaled QPixmap ready for display
        """
        # Scale down large images for UI performance
        h, w = img_array.shape[:2]
        if max(h, w) > PREVIEW_MAX_DIMENSION:
            scale = PREVIEW_MAX_DIMENSION / max(h, w)
            img_array = cv2.resize(img_array, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
        
        # CRITICAL: Memory safety for QImage/numpy interaction
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        
        # Create QImage (shares memory with rgb array)
        q_img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        # Create a DEEP COPY to break memory sharing
        q_img_copy = q_img.copy()
        # Now safe to let rgb be garbage collected
        del rgb, q_img
        
        # Scale and create pixmap from the copy
        return self._scale_pixmap_for_area(q_img_copy, scroll_area)
    
    def _scale_pixmap_for_area(self, q_img: QImage, scroll_area: QScrollArea) -> QPixmap:
        """Scale a QImage to fit within a scroll area.
        
        Args:
            q_img: The QImage to scale.
            scroll_area: The scroll area to fit the image into.
            
        Returns:
            Scaled QPixmap maintaining aspect ratio.
        """
        avail_w = max(scroll_area.width() - PREVIEW_MARGIN, PREVIEW_MIN_WIDTH)
        avail_h = max(scroll_area.height() - PREVIEW_MARGIN, PREVIEW_MIN_HEIGHT)
        scale = min(avail_w / q_img.width(), avail_h / q_img.height(), 1.0)
        new_w = int(q_img.width() * scale)
        new_h = int(q_img.height() * scale)
        
        return QPixmap.fromImage(q_img).scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    def clear(self) -> None:
        """Clear both preview displays."""
        self.prev_label.setText("No image")
        self.prev_stage_label.setText("Previous: -")
        self.curr_label.setText("No image")
        self.curr_stage_label.setText("Current: -")
        self._is_pre_ai = False
    
    def set_original_preview(self, img_array: np.ndarray) -> None:
        """Set the original image as the current preview (for initial load).
        
        Args:
            img_array: Original image in BGR format.
        """
        pixmap = self._create_scaled_pixmap(img_array, self.curr_scroll)
        self.curr_label.setPixmap(pixmap)
        self.curr_stage_label.setText("Current: Original")
        # Clear previous side
        self.prev_label.setText("-")
        self.prev_stage_label.setText("Previous: -")
