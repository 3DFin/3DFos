"""
3DFos package

Main package for 3D forestry segmentation.
"""

# Re-export backend modules for backward compatibility
from three_d_fos.backend import (
    model,
    module,
    registry,
    seghead,
    structure,
    transform,
    utils,
)

__all__ = [
    "model",
    "module",
    "registry",
    "seghead",
    "structure",
    "transform",
    "utils",
]
