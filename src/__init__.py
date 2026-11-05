"""In Silico LLM RecSys package."""

import os

# Prevent OpenMP runtime collision between MKL and PyTorch on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
