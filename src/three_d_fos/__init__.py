"""
3DFos package

Main package for 3D forestry segmentation.
"""

# Re-export backend modules for backward compatibility
from three_d_fos.backend import (
    inference,
    liteptv1m1_model,
    module,
    ptv3v1m1_model,
    registry,
    seghead,
    structure,
    transform,
    utils,
)

__all__ = [
    "inference",
    "liteptv1m1_model",
    "module",
    "ptv3v1m1_model",
    "registry",
    "seghead",
    "structure",
    "transform",
    "utils",
]
