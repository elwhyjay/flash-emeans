"""Fused Triton kernels for IVF assignment in faiss e-means."""
from .patch import enable

__version__ = "0.1.0"
__all__ = ["enable"]
