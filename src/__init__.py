"""AlphaForge package init.

Pre-import ordering fix for Windows: torch and xgboost both ship OpenMP
runtimes (libiomp5md.dll). Loading them in the wrong order segfaults.
We import torch first and set KMP_DUPLICATE_LIB_OK as a safety net.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch  # noqa: F401  (import order matters)
except ImportError:
    pass
