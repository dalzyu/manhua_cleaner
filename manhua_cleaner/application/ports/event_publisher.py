"""Event Publisher port - interface for publishing events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ProcessingEvent:
    """Event during processing."""
    stage: str
    message: str
    progress: float | None = None  # 0.0 to 1.0
    image_path: Path | None = None


@runtime_checkable
class EventPublisher(Protocol):
    """Port for publishing processing events."""
    
    def publish(self, event: ProcessingEvent) -> None:
        """Publish an event."""
        ...
    
    def subscribe(self, callback: Callable[[ProcessingEvent], None]) -> None:
        """Subscribe to events."""
        ...


class SimpleEventPublisher:
    """Simple synchronous event publisher."""
    
    def __init__(self):
        self._subscribers: list[Callable[[ProcessingEvent], None]] = []
    
    def publish(self, event: ProcessingEvent) -> None:
        for callback in self._subscribers:
            callback(event)
    
    def subscribe(self, callback: Callable[[ProcessingEvent], None]) -> None:
        self._subscribers.append(callback)
