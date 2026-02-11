"""Batch processor for processing multiple images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from ...domain.entities.image import Image, ProcessingResult
from ...domain.value_objects.config import ProcessingConfig
from ..ports.event_publisher import EventPublisher, ProcessingEvent, SimpleEventPublisher
from .text_removal import TextRemovalService

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int
    successful: int
    failed: int
    processing_time_ms: float
    results: list[tuple[Path, ProcessingResult]]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total


class BatchProcessor:
    """Process multiple images in batch."""
    
    def __init__(
        self,
        text_removal_service: TextRemovalService,
        event_publisher: EventPublisher | None = None
    ):
        self._service = text_removal_service
        self._events = event_publisher or SimpleEventPublisher()
    
    def process_files(
        self,
        files: list[Path],
        output_dir: Path,
        progress_callback: Callable[[int, int, str], None] | None = None
    ) -> BatchResult:
        """Process multiple files.
        
        Args:
            files: List of image files to process
            output_dir: Directory to save results
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Batch processing result
        """
        import time
        start_time = time.time()
        
        results: list[tuple[Path, ProcessingResult]] = []
        successful = 0
        failed = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._events.publish(ProcessingEvent(
            stage="batch_start",
            message=f"Starting batch of {len(files)} files",
            progress=0.0
        ))
        
        with self._service:
            for i, file_path in enumerate(files, 1):
                if progress_callback:
                    progress_callback(i, len(files), f"Processing {file_path.name}")
                
                self._events.publish(ProcessingEvent(
                    stage="processing",
                    message=f"Processing {file_path.name}",
                    progress=(i - 1) / len(files),
                    image_path=file_path
                ))
                
                try:
                    result = self._service.remove_text(file_path)
                    results.append((file_path, result))
                    
                    if result.success and result.image:
                        output_file = output_dir / f"cleaned_{file_path.name}"
                        result.image.save(output_file)
                        successful += 1
                    else:
                        failed += 1
                        logger.error(f"Failed {file_path.name}: {result.error_message}")
                        
                except Exception as e:
                    logger.exception(f"Error processing {file_path.name}")
                    failed += 1
                    results.append((file_path, ProcessingResult.failure(str(e))))
        
        elapsed = (time.time() - start_time) * 1000
        
        self._events.publish(ProcessingEvent(
            stage="batch_complete",
            message=f"Batch complete: {successful}/{len(files)} succeeded",
            progress=1.0
        ))
        
        return BatchResult(
            total=len(files),
            successful=successful,
            failed=failed,
            processing_time_ms=elapsed,
            results=results
        )
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
        **kwargs
    ) -> BatchResult:
        """Process all images in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: File extensions to process
            **kwargs: Additional args for process_files
            
        Returns:
            Batch processing result
        """
        files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        files.sort()
        
        return self.process_files(files, output_dir, **kwargs)
    
    def subscribe_to_events(self, callback: Callable[[ProcessingEvent], None]) -> None:
        """Subscribe to processing events."""
        self._events.subscribe(callback)
