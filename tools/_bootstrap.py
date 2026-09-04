"""Make the src-layout package importable for direct tool execution."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    """Add the repository src directory and return the project root."""
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return project_root
