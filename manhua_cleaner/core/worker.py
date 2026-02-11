"""Multiprocessing worker pool for OCR tasks."""

import logging
import multiprocessing as mp
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .ocr_factory import OCRFactory
from .image_ops import Contour, contour_to_rect

logger = logging.getLogger(__name__)


@dataclass
class OCRTask:
    """Task for OCR processing."""
    image_path: Path
    index: int
    total: int


@dataclass
class OCRResult:
    """Result of OCR processing."""
    success: bool
    image_path: Path
    boxes: Optional[list[Contour]] = None
    text_regions: Optional[list] = None  # List of TextRegion dicts for whitelist filtering
    error_message: Optional[str] = None
    index: int = 0
    total: int = 0


def ocr_worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    ocr_model: str,
    pipeline_version: str,
    use_tensorrt: bool,
    precision: str
) -> None:
    """Worker process that handles OCR tasks.
    
    This runs in a separate process to keep the main process responsive
    and allow model reuse across multiple images.
    
    Args:
        worker_id: Unique identifier for this worker
        task_queue: Queue for receiving tasks
        result_queue: Queue for sending results
        stop_event: Event to signal shutdown
        ocr_model: OCR backend type ('paddleocr' or 'easyocr')
        pipeline_version: PaddleOCR pipeline version
        use_tensorrt: Use TensorRT acceleration if available
        precision: OCR model precision (fp16 or fp32)
    """
    # Initialize logging for subprocess
    # Use separate log file to avoid conflicts with main process
    log_file = f"manhua_cleaner_worker_{worker_id}.log"
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    except (OSError, PermissionError):
        pass  # Fall back to console-only
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Worker{worker_id} - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Replace any existing handlers
    )
    worker_logger = logging.getLogger("ocr_worker")
    worker_logger.info(f"Worker {worker_id} started (PID: {os.getpid()})")
    
    # Load model once
    try:
        spotter = OCRFactory.create(
            model_type=ocr_model,
            pipeline_version=pipeline_version,
            use_tensorrt=use_tensorrt,
            precision=precision
        )
        spotter.load()
        worker_logger.info(f"Worker {worker_id}: OCR model loaded ({ocr_model})")
    except Exception as e:
        worker_logger.error(f"Worker {worker_id}: Failed to load OCR model: {e}")
        result_queue.put({
            "type": "fatal_error",
            "worker_id": worker_id,
            "message": f"Failed to load OCR model: {e}"
        })
        return
    
    # Track processed count for periodic cleanup
    processed_count = 0
    
    # Process tasks
    while not stop_event.is_set():
        try:
            task_data = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # Normal timeout, continue polling
        except (OSError, IOError) as e:
            worker_logger.error(f"Worker {worker_id}: Queue communication error: {e}")
            continue
        
        if task_data is None or task_data == "STOP":
            worker_logger.info(f"Worker {worker_id}: Received stop signal")
            break
        
        # Parse task
        try:
            # Use absolute path to avoid working directory issues in subprocess
            image_path = Path(task_data["image_path"]).resolve()
            index = task_data["index"]
            total = task_data["total"]
            worker_logger.debug(f"Worker {worker_id}: Processing {image_path.name}")
            
            # Verify image exists before processing
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Run OCR with text detection (for whitelist filtering)
            text_regions = spotter.detect_with_text(image_path)
            
            if not text_regions:
                result_queue.put({
                    "type": "success",
                    "worker_id": worker_id,
                    "image_path": str(image_path),
                    "boxes": [],
                    "text_regions": [],
                    "index": index,
                    "total": total
                })
                processed_count += 1
                continue
            
            # Convert to quadrilaterals (raw OCR rectangles only)
            quads = [contour_to_rect(r.contour) for r in text_regions]
            
            # Convert to list format for serialization
            boxes_list = []
            for quad in quads:
                contour = quad.reshape(-1, 1, 2)
                boxes_list.append(contour.tolist())
            
            # Serialize text regions for whitelist filtering in main thread
            text_regions_data = []
            for r in text_regions:
                text_regions_data.append({
                    "contour": r.contour.tolist(),
                    "text": r.text,
                    "confidence": r.confidence
                })
            
            result_queue.put({
                "type": "success",
                "worker_id": worker_id,
                "image_path": str(image_path),
                "boxes": boxes_list,
                "text_regions": text_regions_data,
                "index": index,
                "total": total
            })
            
            processed_count += 1
            
            # Periodic cleanup every 10 images
            if processed_count % 10 == 0:
                import gc
                gc.collect()
            
        except Exception as e:
            worker_logger.error(f"Worker {worker_id}: Error processing {task_data.get('image_path', 'unknown')}: {e}")
            result_queue.put({
                "type": "error",
                "worker_id": worker_id,
                "image_path": task_data.get("image_path", ""),
                "error": str(e),
                "index": task_data.get("index", 0),
                "total": task_data.get("total", 0)
            })
            processed_count += 1
    
    # Cleanup
    worker_logger.info(f"Worker {worker_id}: Unloading model and exiting")
    spotter.unload()
    
    import gc
    gc.collect()


class OCRWorkerPool:
    """Pool for managing multiple OCR worker processes.
    
    This class manages a pool of worker processes for parallel OCR processing.
    The pool size defaults to the number of CPU cores but can be configured.
    
    Example:
        with OCRWorkerPool(num_workers=4) as pool:
            for i, path in enumerate(image_paths):
                pool.submit(path, i, len(image_paths))
            
            results = []
            while len(results) < len(image_paths):
                result = pool.get_result(timeout=1.0)
                if result:
                    results.append(result)
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        ocr_model: str = "paddleocr",
        ocr_pipeline_version: str = "v1.5",
        ocr_use_tensorrt: bool = True,
        ocr_precision: str = "fp16"
    ):
        """Initialize worker pool.
        
        Args:
            num_workers: Number of worker processes. Defaults to 1 to avoid excessive VRAM use.
                Each worker loads a full OCR model (~16GB VRAM), so increase with caution.
            ocr_model: OCR backend type ('paddleocr' or 'easyocr')
            ocr_pipeline_version: PaddleOCR pipeline version
            ocr_use_tensorrt: Use TensorRT acceleration if available
            ocr_precision: OCR model precision (fp16 or fp32)
        """
        if num_workers is None:
            self.num_workers = 1  # Default to 1 to avoid excessive VRAM usage
        else:
            self.num_workers = max(1, num_workers)
        self.ocr_model = ocr_model
        self.ocr_pipeline_version = ocr_pipeline_version
        self.ocr_use_tensorrt = ocr_use_tensorrt
        self.ocr_precision = ocr_precision
        self._task_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._workers: list[mp.Process] = []
        self._pending_tasks = 0
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the worker processes."""
        if self._workers and any(w.is_alive() for w in self._workers):
            logger.debug("Worker pool already started")
            return
        
        # Use a larger queue to prevent submission blocking with many files
        # Each worker can have multiple tasks queued for better throughput
        self._task_queue = mp.Queue(maxsize=max(100, self.num_workers * 4))
        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()
        self._workers = []
        self._pending_tasks = 0
        
        for i in range(self.num_workers):
            worker = mp.Process(
                target=ocr_worker_process,
                args=(
                    i,
                    self._task_queue,
                    self._result_queue,
                    self._stop_event,
                    self.ocr_model,
                    self.ocr_pipeline_version,
                    self.ocr_use_tensorrt,
                    self.ocr_precision
                ),
                daemon=True,
                name=f"OCRWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"OCR worker pool started with {self.num_workers} workers")
    
    def submit(self, image_path: Path, index: int, total: int) -> None:
        """Submit a task to the pool.
        
        Tasks are distributed round-robin to available workers via the queue.
        
        Args:
            image_path: Path to image
            index: Index in batch
            total: Total number of images
            
        Raises:
            RuntimeError: If pool is not started
        """
        if self._task_queue is None:
            raise RuntimeError("Worker pool not started")
        
        with self._lock:
            self._pending_tasks += 1
        
        # Block until queue has space (backpressure)
        self._task_queue.put({
            "image_path": str(image_path),
            "index": index,
            "total": total
        }, block=True)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[OCRResult]:
        """Get a result from the pool.
        
        Args:
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            OCR result or None if timeout
            
        Raises:
            RuntimeError: If pool is not started or worker encountered fatal error
        """
        if self._result_queue is None:
            raise RuntimeError("Worker pool not started")
        
        try:
            data = self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except (OSError, IOError) as e:
            logger.error(f"Queue error: {e}")
            raise RuntimeError(f"Result queue error: {e}") from e
        
        with self._lock:
            self._pending_tasks = max(0, self._pending_tasks - 1)
        
        # Handle fatal worker errors
        if data.get("type") == "fatal_error":
            raise RuntimeError(
                f"Worker {data.get('worker_id')} fatal error: {data.get('message')}"
            )
        
        # Validate expected fields
        if "type" not in data:
            raise RuntimeError(f"Malformed result from worker: {data}")
        
        boxes = None
        if "boxes" in data and data["boxes"] is not None:
            try:
                boxes = [np.array(b, dtype=np.int32) for b in data["boxes"]]
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to parse boxes from worker: {e}")
                boxes = None
        
        # Parse text regions for whitelist filtering
        text_regions = None
        if "text_regions" in data and data["text_regions"] is not None:
            try:
                from .text_region import TextRegion
                text_regions = []
                for r_data in data["text_regions"]:
                    text_regions.append(TextRegion(
                        contour=np.array(r_data["contour"], dtype=np.int32),
                        text=r_data["text"],
                        confidence=r_data["confidence"]
                    ))
            except (TypeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse text_regions from worker: {e}")
                text_regions = None
        
        return OCRResult(
            success=data.get("type") == "success",
            image_path=Path(data.get("image_path", "")),
            boxes=boxes,
            text_regions=text_regions,
            error_message=data.get("error"),
            index=data.get("index", 0),
            total=data.get("total", 0)
        )
    
    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        with self._lock:
            return self._pending_tasks
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop all worker processes gracefully.
        
        Sends stop signal to all workers and waits for them to finish.
        If workers don't stop within timeout, they are terminated.
        
        Args:
            timeout: Timeout in seconds for graceful shutdown per worker
        """
        if not self._workers:
            return
        
        logger.info("Stopping OCR worker pool...")
        
        # Signal stop
        if self._stop_event is not None:
            self._stop_event.set()
        
        # Send stop message to each worker
        if self._task_queue is not None:
            for _ in self._workers:
                try:
                    self._task_queue.put("STOP", block=False)
                except Exception:
                    pass
        
        # Wait for workers to finish
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=timeout)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not stop gracefully, terminating")
                    worker.terminate()
                    worker.join(timeout=1.0)
        
        self._workers = []
        logger.info("OCR worker pool stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Backwards compatibility - single worker mode
class SingleOCRWorker(OCRWorkerPool):
    """Single worker mode for backwards compatibility.
    
    Deprecated: Use OCRWorkerPool with num_workers=1 instead.
    """
    
    def __init__(self, expand_pixels: int = 25):
        """Initialize single worker."""
        super().__init__(num_workers=1)
        logger.warning(
            "SingleOCRWorker is deprecated; use OCRWorkerPool with num_workers=1. "
            "expand_pixels is ignored."
        )
