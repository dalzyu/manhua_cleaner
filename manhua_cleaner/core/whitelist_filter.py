"""Whitelist filtering for text regions based on character patterns."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .text_region import Textbox

# Import from config to avoid duplication
from ..config import WHITELIST_PRESETS

logger = logging.getLogger(__name__)


@dataclass
class WhitelistConfig:
    """Configuration for whitelist filtering."""
    
    enabled: bool = False
    patterns: list[str] = field(default_factory=list)
    group_distance: float = 50.0
    
    @classmethod
    def from_preset(cls, preset_name: str, group_distance: float = 50.0) -> "WhitelistConfig":
        """Create config from a preset name."""
        patterns = WHITELIST_PRESETS.get(preset_name, [])
        return cls(
            enabled=True,
            patterns=patterns.copy(),
            group_distance=group_distance
        )
    
    @classmethod
    def from_file(cls, filepath: Path | str) -> "WhitelistConfig":
        """Load config from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Whitelist file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            enabled=data.get('enabled', True),
            patterns=data.get('patterns', []),
            group_distance=data.get('group_distance', 50.0)
        )
    
    def to_file(self, filepath: Path | str) -> None:
        """Save config to JSON file."""
        filepath = Path(filepath)
        data = {
            'enabled': self.enabled,
            'patterns': self.patterns,
            'group_distance': self.group_distance
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_text_file(cls, filepath: Path | str, group_distance: float = 50.0) -> "WhitelistConfig":
        """
        Load patterns from a text file (one pattern per line).
        
        Lines starting with # are treated as comments.
        Empty lines are ignored.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Whitelist file not found: {filepath}")
        
        patterns = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                patterns.append(line)
        
        return cls(
            enabled=True,
            patterns=patterns,
            group_distance=group_distance
        )


class WhitelistFilter:
    """Filter text regions based on whitelist patterns."""
    
    def __init__(self, patterns: list[str]):
        """
        Initialize the whitelist filter.
        
        Args:
            patterns: List of regex patterns. If any pattern matches,
                     the text is whitelisted (not cleaned).
        """
        self.patterns = patterns
        self._compiled_patterns: list[re.Pattern] = []
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns."""
        self._compiled_patterns = []
        for pattern in self.patterns:
            try:
                compiled = re.compile(pattern)
                self._compiled_patterns.append(compiled)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    
    def is_whitelisted(self, text: str) -> bool:
        """
        Check if text matches any whitelist pattern.
        
        Args:
            text: The text to check.
            
        Returns:
            True if text should be kept (whitelisted), False if should be cleaned.
        """
        text = text.strip()
        if not text:
            return False  # Empty text is not whitelisted
        
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def filter_textboxes(self, textboxes: list["Textbox"]) -> tuple[list["Textbox"], list["Textbox"]]:
        """
        Separate textboxes into whitelisted and to-clean lists.
        
        Args:
            textboxes: List of textboxes to filter.
            
        Returns:
            Tuple of (whitelisted_textboxes, textboxes_to_clean).
        """
        whitelisted = []
        to_clean = []
        
        for textbox in textboxes:
            if self.is_whitelisted(textbox.combined_text):
                whitelisted.append(textbox)
                logger.debug(f"Whitelisted textbox: '{textbox.combined_text[:50]}...'")
            else:
                to_clean.append(textbox)
        
        return whitelisted, to_clean
    
    def filter_regions(self, textboxes: list["Textbox"]) -> list["TextRegion"]:
        """
        Get regions that should be cleaned (excluding whitelisted).
        
        Args:
            textboxes: List of textboxes to filter.
            
        Returns:
            List of regions from non-whitelisted textboxes.
        """
        _, to_clean = self.filter_textboxes(textboxes)
        
        regions = []
        for textbox in to_clean:
            regions.extend(textbox.regions)
        
        return regions
    
    def get_stats(self, textboxes: list["Textbox"]) -> dict:
        """
        Get statistics about whitelist filtering.
        
        Args:
            textboxes: List of all textboxes.
            
        Returns:
            Dict with statistics.
        """
        whitelisted, to_clean = self.filter_textboxes(textboxes)
        
        total_regions = sum(len(tb.regions) for tb in textboxes)
        whitelisted_regions = sum(len(tb.regions) for tb in whitelisted)
        
        return {
            'total_textboxes': len(textboxes),
            'whitelisted_textboxes': len(whitelisted),
            'textboxes_to_clean': len(to_clean),
            'total_regions': total_regions,
            'whitelisted_regions': whitelisted_regions,
            'regions_to_clean': total_regions - whitelisted_regions,
        }


def create_filter_from_config(config: WhitelistConfig) -> WhitelistFilter | None:
    """
    Create a WhitelistFilter from configuration.
    
    Args:
        config: Whitelist configuration.
        
    Returns:
        WhitelistFilter if enabled and has patterns, None otherwise.
    """
    if not config.enabled or not config.patterns:
        return None
    
    return WhitelistFilter(config.patterns)
