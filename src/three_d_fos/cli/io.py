from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData


def read_ply(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a PLY file and extract point cloud and scalar attributes."""
    with open(filepath, "rb") as f:
        cloud = PlyData.read(f)

    vertices = cloud["vertex"]
    xyz = np.vstack((vertices["x"], vertices["y"], vertices["z"])).T

    dist_axes = vertices["scalar_dist_axes"]
    z0 = vertices["scalar_Z0"]

    if not np.all(np.isfinite(dist_axes)) or not np.all(np.isfinite(z0)):
        raise ValueError("Inf values detected in scalar fields.")

    return xyz, z0, dist_axes


def read_las(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a LAS file and extract point cloud and scalar attributes."""
    las = laspy.read(filepath)
    xyz = np.vstack((las.x, las.y, las.z)).T

    if not hasattr(las, "dist_axes") or not hasattr(las, "Z0"):
        raise ValueError("LAS file missing required scalar fields: 'dist_axes' and 'Z0'.")

    z0 = las.Z0
    dist_axes = las.dist_axes

    if not np.all(np.isfinite(dist_axes)) or not np.all(np.isfinite(z0)):
        raise ValueError("Non-finite values detected in scalar fields.")

    return xyz, z0, dist_axes


def write_las_predictions(
    output_path: str,
    original_coord: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Write classification labels as a LAS file with point coordinates."""
    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)
    las.x, las.y, las.z = original_coord.T
    las.classification = labels
    las.write(output_path)
