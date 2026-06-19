from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointCloudData:
    """Container for point cloud data with required attributes."""

    xyz: np.ndarray  # Nx3 array of point coordinates
    z0: np.ndarray  # N array of elevation values
    dist_axes: np.ndarray  # N array of distance to axis values
    source_name: str = ""


@dataclass
class SegmentationResult:
    """Container for segmentation results."""

    original_coord: np.ndarray  # Nx3 array of original point coordinates
    labels: np.ndarray  # N array of classification labels


class PointCloudSource(ABC):
    """Abstract base class for point cloud data sources."""

    @abstractmethod
    def load(self) -> PointCloudData:
        """Load and return point cloud data."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this source."""
        raise NotImplementedError


class FilePointCloudSource(PointCloudSource):
    """Point cloud source from file (PLY/LAS/LAZ)."""

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def get_name(self) -> str:
        return self.filepath.name

    def load(self) -> PointCloudData:
        """Load point cloud from file."""
        suffix = self.filepath.suffix.lower()

        if suffix == ".ply":
            return self._load_ply()
        elif suffix in (".las", ".laz"):
            return self._load_las()
        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. Supported: .ply, .las, .laz")

    def _load_ply(self) -> PointCloudData:
        """Load PLY file."""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError("plyfile is required for PLY support. Install with: pip install plyfile")

        with open(self.filepath, "rb") as f:
            cloud = PlyData.read(f)

        vertices = cloud["vertex"]
        xyz = np.vstack((vertices["x"], vertices["y"], vertices["z"])).T
        dist_axes = vertices["scalar_dist_axes"]
        z0 = vertices["scalar_Z0"]

        if not np.all(np.isfinite(dist_axes)) or not np.all(np.isfinite(z0)):
            raise ValueError("Inf values detected in scalar fields.")

        return PointCloudData(xyz=xyz, z0=z0, dist_axes=dist_axes, source_name=self.get_name())

    def _load_las(self) -> PointCloudData:
        """Load LAS/LAZ file."""
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LAS/LAZ support. Install with: pip install laspy")

        las = laspy.read(self.filepath)
        xyz = np.vstack((las.x, las.y, las.z)).T

        if not hasattr(las, "dist_axes") or not hasattr(las, "Z0"):
            raise ValueError("LAS file missing required scalar fields: 'dist_axes' and 'Z0'.")

        z0 = las.Z0
        dist_axes = las.dist_axes

        if not np.all(np.isfinite(dist_axes)) or not np.all(np.isfinite(z0)):
            raise ValueError("Non-finite values detected in scalar fields.")

        return PointCloudData(xyz=xyz, z0=z0, dist_axes=dist_axes, source_name=self.get_name())


class PointCloudDestination(ABC):
    """Abstract base class for point cloud output destinations."""

    @abstractmethod
    def save(self, result: SegmentationResult) -> None:
        """Save segmentation results to this destination."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this destination."""
        raise NotImplementedError


class FilePointCloudDestination(PointCloudDestination):
    """Save segmentation results to a LAS file."""

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def get_name(self) -> str:
        return str(self.filepath)

    def save(self, result: SegmentationResult) -> None:
        """Write classification labels as a LAS file with point coordinates."""
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LAS output. Install with: pip install laspy")

        header = laspy.LasHeader(version="1.4", point_format=6)
        las = laspy.LasData(header)
        las.x, las.y, las.z = result.original_coord.T
        las.classification = result.labels
        las.write(str(self.filepath))
