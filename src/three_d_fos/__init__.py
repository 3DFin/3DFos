"""
3DFos package

Main package for 3D forestry segmentation.
"""

# Import version
from three_d_fos.__version__ import __version__

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
    "__version__",
    "model",
    "module",
    "registry",
    "seghead",
    "structure",
    "transform",
    "utils",
]
