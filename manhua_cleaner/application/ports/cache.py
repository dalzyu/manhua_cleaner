"""Cache port - interface for caching."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar('T')


@runtime_checkable
class Cache(Protocol):
    """Port for caching operations."""
    
    def get(self, key: str) -> object | None:
        """Get value from cache."""
        ...
    
    def set(self, key: str, value: object, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL (seconds)."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cached values."""
        ...
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class MemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, max_size: int = 100):
        self._data: dict[str, object] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> object | None:
        return self._data.get(key)
    
    def set(self, key: str, value: object, ttl: int | None = None) -> None:
        # Simple LRU: if at capacity, remove arbitrary item
        if len(self._data) >= self._max_size and key not in self._data:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value
    
    def delete(self, key: str) -> None:
        self._data.pop(key, None)
    
    def clear(self) -> None:
        self._data.clear()
    
    def has(self, key: str) -> bool:
        return key in self._data
