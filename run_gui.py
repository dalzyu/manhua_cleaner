#!/usr/bin/env python3
"""Launcher for the Manhua Image Cleaner GUI (PyQt6)."""

import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    from manhua_cleaner.gui import main
    sys.exit(main())
