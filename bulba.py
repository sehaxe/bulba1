#!/usr/bin/env python3
"""Bulba 1 - Entry point"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bulba1.cli import main

if __name__ == "__main__":
    main()
