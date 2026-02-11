"""Processing controller for managing the background processing thread."""

import logging
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from ..core import BatchProcessor, OCRWorkerPool, ProcessingConfig
from ..core.text_region import TextRegion
from ..exceptions import (
    ValidationError,
    ImageProcessingError,
    OCRError,
    ModelError,
)

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Thread for running the image processing pipeline."""
    
    # Signals for thread-safe UI updates
    log_signal = pyqtSignal(str, str)  # message, level
    status_signal = pyqtSignal(str)  # status text
    progress_signal = pyqtSignal(int, int)  # current, total
    file_progress_signal = pyqtSignal(int, int, str)  # current, total, filename
    preview_signal = pyqtSignal(object, str, bool)  # image data, stage label, is_pre_ai
    stats_signal = pyqtSignal(int, int)  # smart_filled, ai_processed
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(
        self,
        processor: BatchProcessor,
        files: list[Path],
        output_dir: Path,
        debug_mode: bool = False
    ):
        super().__init__()
        self.processor = processor
        self.files = files
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.is_running = True
        self._stop_event = None
    
    def stop(self) -> None:
        """Request the thread to stop."""
        self.is_running = False
        if self._stop_event:
            self._stop_event.set()
    
    def run(self) -> None:
        """Main processing logic."""
        failed_images = []
        ocr_cache = {}  # path -> boxes (contours)
        text_regions_cache = {}  # path -> list[TextRegion] (for whitelist filtering)
        
        # Timing tracking
        total_start_time = time.time()
        ocr_time = 0.0
        cleaning_time = 0.0
        
        try:
            # Phase 1: OCR with worker pool
            self.log_signal.emit("Phase 1: Running OCR on all images...", "info")
            self.status_signal.emit(f"OCR: Processing {len(self.files)} images...")
            ocr_start_time = time.time()
            
            with OCRWorkerPool(
                num_workers=self.processor.config.ocr_workers,
                ocr_model=self.processor.config.ocr_model,
                ocr_pipeline_version=self.processor.config.ocr_pipeline_version,
                ocr_use_tensorrt=self.processor.config.ocr_use_tensorrt,
                ocr_precision=self.processor.config.ocr_precision
            ) as pool:
                # Interleave submission and result collection to avoid queue overflow
                pending = len(self.files)
                completed = 0
                submit_index = 0
                
                # Submit first batch (up to queue capacity)
                initial_batch = min(pending, 20)  # Submit first 20 without blocking
                for i in range(initial_batch):
                    if not self.is_running:
                        self.log_signal.emit("Processing cancelled", "warning")
                        return
                    pool.submit(self.files[i], i, len(self.files))
                    submit_index += 1
                
                # Collect results and submit remaining tasks
                while pending > 0 and self.is_running:
                    result = pool.get_result(timeout=0.5)
                    if result:
                        pending -= 1
                        completed += 1
                        if result.success and result.boxes is not None:
                            ocr_cache[result.image_path] = result.boxes
                            text_regions_cache[result.image_path] = result.text_regions
                            self.status_signal.emit(
                                f"OCR: {result.image_path.name} ({result.index+1}/{result.total})"
                            )
                            self.log_signal.emit(
                                f"OCR complete: {result.image_path.name}", "success"
                            )
                            # Progress: 0-50% for OCR phase
                            progress = int(completed / len(self.files) * 50)
                            self.progress_signal.emit(progress, 100)
                            
                            if self.debug_mode and result.boxes:
                                self.log_signal.emit(
                                    f"  [DEBUG] Found {len(result.boxes)} text region(s)", "info"
                                )
                        else:
                            failed_images.append((result.image_path.name, result.error_message))
                            self.log_signal.emit(
                                f"OCR failed: {result.image_path.name}: {result.error_message}", 
                                "error"
                            )
                        
                        # Submit next task if available (keeps queue populated)
                        if submit_index < len(self.files):
                            if not self.is_running:
                                self.log_signal.emit("Processing cancelled", "warning")
                                return
                            pool.submit(self.files[submit_index], submit_index, len(self.files))
                            submit_index += 1
            
            if not self.is_running:
                return
            
            # Record OCR phase completion time
            ocr_time = time.time() - ocr_start_time
            
            if not ocr_cache:
                self.log_signal.emit("No successful OCR results to process", "warning")
                self.finished_signal.emit(False, "No OCR results")
                return
            
            # Apply whitelist filtering if enabled (ONCE, using worker results)
            if self.processor.config.whitelist_enabled:
                self.log_signal.emit("Applying whitelist filtering...", "info")
                from ..core.textbox_grouper import TextboxGrouper
                from ..core.whitelist_filter import WhitelistFilter, create_filter_from_config, WhitelistConfig
                
                # Log what patterns are being used
                patterns = self.processor.config.whitelist_patterns
                self.log_signal.emit(f"  Using {len(patterns)} whitelist pattern(s):", "info")
                for i, p in enumerate(patterns, 1):
                    self.log_signal.emit(f"    Pattern {i}: {p}", "debug")
                
                whitelist_config = WhitelistConfig(
                    enabled=True,
                    patterns=patterns,
                    group_distance=self.processor.config.whitelist_group_distance
                )
                filter_ = create_filter_from_config(whitelist_config)
                
                if filter_ is not None:
                    for image_path in list(ocr_cache.keys()):
                        text_regions = text_regions_cache.get(image_path, [])
                        num_regions = len(text_regions) if text_regions else 0
                        self.log_signal.emit(f"  {image_path.name}: {num_regions} text region(s) from OCR", "info")
                        
                        if text_regions:
                            # Log what text was detected
                            self.log_signal.emit("  Detected text:", "debug")
                            for j, r in enumerate(text_regions):
                                self.log_signal.emit(f"    Region {j+1}: '{r.text}'", "debug")
                            
                            grouper = TextboxGrouper(max_distance=self.processor.config.whitelist_group_distance)
                            textboxes = grouper.group_regions(text_regions)
                            
                            self.log_signal.emit(f"  Grouped into {len(textboxes)} textbox(es):", "debug")
                            for j, tb in enumerate(textboxes):
                                is_whitelisted = filter_.is_whitelisted(tb.combined_text)
                                status = "KEEP" if is_whitelisted else "CLEAN"
                                self.log_signal.emit(
                                    f"    Textbox {j+1}: '{tb.combined_text}' -> {status}", 
                                    "debug"
                                )
                            
                            stats = filter_.get_stats(textboxes)
                            
                            self.log_signal.emit(
                                f"  {image_path.name}: {stats['total_textboxes']} textbox(es), "
                                f"{stats['whitelisted_textboxes']} whitelisted, "
                                f"{stats['textboxes_to_clean']} to clean",
                                "info"
                            )
                            
                            # Filter and update cache with only contours to clean
                            filtered_regions = filter_.filter_regions(textboxes)
                            ocr_cache[image_path] = [r.contour for r in filtered_regions]
                        else:
                            self.log_signal.emit(f"  {image_path.name}: No text regions to filter", "warning")
            
            # Phase 2: Image generation
            self.log_signal.emit("Phase 2: Generating cleaned images...", "info")
            self.status_signal.emit("Loading image model...")
            cleaning_start_time = time.time()
            
            # Pre-load OCR cache
            self.log_signal.emit(f"Caching OCR results for {len(ocr_cache)} image(s)...", "debug")
            for path, boxes in ocr_cache.items():
                self.processor._ocr_cache.put(path, boxes)
                self.log_signal.emit(f"  Cached: {path.name} ({len(boxes)} boxes)", "debug")
            
            success_count = 0
            
            for i, file_path in enumerate(self.files):
                if not self.is_running:
                    self.log_signal.emit("Processing cancelled", "warning")
                    return
                
                if file_path not in ocr_cache:
                    continue
                
                filename = file_path.name
                self.status_signal.emit(f"Processing: {filename} ({i+1}/{len(self.files)})")
                self.log_signal.emit(f"Processing: {filename}...", "info")
                
                # Emit file progress for UI counter
                self.file_progress_signal.emit(i + 1, len(self.files), filename)
                
                # Progress: 50-100% for Generation phase
                progress = 50 + int((i + 1) / len(self.files) * 50)
                self.progress_signal.emit(progress, 100)
                
                try:
                    def preview_callback(img, stage, is_pre_ai=False):
                        """Emit preview signal with optionally scaled-down image."""
                        # Scale down large images for preview to reduce UI thread load
                        h, w = img.shape[:2]
                        if max(h, w) > 1200:  # PREVIEW_MAX_DIMENSION
                            scale = 1200 / max(h, w)
                            img = cv2.resize(img, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA)
                        
                        if is_pre_ai:
                            self.preview_signal.emit(img.copy(), "Before AI", True)
                        else:
                            self.preview_signal.emit(img.copy(), stage, False)
                    
                    import cv2
                    import numpy as np
                    from PIL import Image
                    
                    def progress_callback(msg: str):
                        """Emit progress messages to log."""
                        self.log_signal.emit(msg, "info")
                    
                    result = self.processor.process_image(
                        file_path, 
                        progress_callback=progress_callback,
                        preview_callback=preview_callback
                    )
                    
                    # process_image raises on failure, so if we get here, it succeeded
                    # Log stats
                    if result.boxes_smart_filled > 0 or result.boxes_processed > 0:
                        self.log_signal.emit(
                            f"  Smart filled: {result.boxes_smart_filled}, "
                            f"AI processed: {result.boxes_processed}", 
                            "info"
                        )
                        self.stats_signal.emit(
                            result.boxes_smart_filled, 
                            result.boxes_processed
                        )
                    
                    # Final preview update - show on right side (current)
                    preview_np = cv2.cvtColor(np.array(result.image), cv2.COLOR_RGB2BGR)
                    self.preview_signal.emit(preview_np, "Final Result", False)
                    
                    # Save output
                    output_path = self.output_dir / f"cleaned_{filename}"
                    result.image.save(output_path)
                    self.log_signal.emit(f"Saved: cleaned_{filename}", "success")
                    success_count += 1
                
                except (ValidationError, ImageProcessingError, OCRError, ModelError) as e:
                    # Expected processing errors - log and continue with next image
                    error_msg = str(e)
                    self.log_signal.emit(f"Processing failed: {filename}: {error_msg}", "error")
                    failed_images.append((filename, error_msg))
                
                except Exception as e:
                    # Unexpected errors - log full traceback
                    import traceback
                    error_msg = f"Unexpected error: {type(e).__name__}: {e}"
                    self.log_signal.emit(f"Processing failed: {filename}: {error_msg}", "error")
                    if self.debug_mode:
                        tb = traceback.format_exc()
                        self.log_signal.emit(f"Traceback:\n{tb}", "debug")
                    failed_images.append((filename, error_msg))
            
            # Record cleaning phase completion time
            cleaning_time = time.time() - cleaning_start_time
            total_time = time.time() - total_start_time
            
            # Format timing summary
            def format_duration(seconds: float) -> str:
                """Format duration in human-readable form."""
                if seconds < 60:
                    return f"{seconds:.1f}s"
                minutes = int(seconds // 60)
                secs = seconds % 60
                if minutes < 60:
                    return f"{minutes}m {secs:.0f}s"
                hours = minutes // 60
                mins = minutes % 60
                return f"{hours}h {mins}m {secs:.0f}s"
            
            timing_summary = (
                f"\n{'='*50}\n"
                f"TIMING SUMMARY\n"
                f"{'='*50}\n"
                f"  OCR Phase:           {format_duration(ocr_time):>12}\n"
                f"  Cleaning Phase:      {format_duration(cleaning_time):>12}\n"
                f"  {'─'*48}\n"
                f"  Total Time:          {format_duration(total_time):>12}\n"
                f"{'='*50}"
            )
            self.log_signal.emit(timing_summary, "info")
            
            # Completion
            if failed_images:
                self.log_signal.emit(
                    f"Completed with errors: {success_count}/{len(self.files)} succeeded", 
                    "warning"
                )
                self.finished_signal.emit(False, f"Completed with {len(failed_images)} errors")
            else:
                self.log_signal.emit(
                    f"All {len(self.files)} images processed successfully!", 
                    "success"
                )
                self.finished_signal.emit(True, f"Processed {len(self.files)} images successfully!")
        
        except Exception as e:
            error_msg = f"Fatal error: {str(e)}"
            self.log_signal.emit(error_msg, "error")
            self.finished_signal.emit(False, error_msg)


class ProcessingController:
    """Controller for managing the processing lifecycle."""
    
    def __init__(self):
        """Initialize processing controller."""
        self.processor: BatchProcessor | None = None
        self.processing_thread: ProcessingThread | None = None
        self._total_smart_filled: int = 0
        self._total_ai_processed: int = 0
        self._on_finished_callbacks: list[Callable[[bool, str], None]] = []
    
    def start_processing(
        self,
        config: ProcessingConfig,
        files: list[Path],
        output_dir: Path,
        debug_mode: bool = False
    ) -> ProcessingThread:
        """Start processing with the given configuration.
        
        Args:
            config: Processing configuration
            files: List of files to process
            output_dir: Output directory for processed images
            debug_mode: Enable debug logging
            
        Returns:
            The processing thread instance
        """
        self.processor = BatchProcessor(config)
        self._total_smart_filled = 0
        self._total_ai_processed = 0
        
        self.processing_thread = ProcessingThread(
            processor=self.processor,
            files=files,
            output_dir=output_dir,
            debug_mode=debug_mode
        )
        
        return self.processing_thread
    
    def stop_processing(self) -> None:
        """Request processing to stop."""
        if self.processing_thread:
            self.processing_thread.stop()
    
    def update_stats(self, smart_filled: int, ai_processed: int) -> None:
        """Update running statistics.
        
        Args:
            smart_filled: Number of boxes smart filled
            ai_processed: Number of boxes AI processed
        """
        self._total_smart_filled += smart_filled
        self._total_ai_processed += ai_processed
    
    def get_stats(self) -> tuple[int, int]:
        """Get current statistics.
        
        Returns:
            Tuple of (total_smart_filled, total_ai_processed)
        """
        return self._total_smart_filled, self._total_ai_processed
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait(2000)
        
        if self.processor:
            self.processor.unload_models()
            self.processor = None
    
    def is_processing(self) -> bool:
        """Check if processing is currently active.
        
        Returns:
            True if processing thread is running
        """
        return (self.processing_thread is not None and 
                self.processing_thread.isRunning())
