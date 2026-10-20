"""Compatibility package for the HiSE source tree.

This project stores the importable Python package under ``hise/src/hise``.  To make
``import hise`` and ``import hise.config`` work directly from the repository root,
we expose the source directory through this package's ``__path__``.
"""

from __future__ import annotations

from pathlib import Path

_package_root = Path(__file__).resolve().parent / "src" / "hise"
__path__ = [str(_package_root), *list(__path__)]
__version__ = "0.1.0"
