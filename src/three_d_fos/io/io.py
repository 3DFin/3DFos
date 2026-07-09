from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData

from three_d_fos.core.feature import Feature


@dataclass
class PointCloudData:
    """Container for point cloud data."""

    xyz: np.ndarray  # Nx3 array of point coordinates
    features: np.ndarray | None
    source_name: str = ""


@dataclass
class SegmentationResult:
    """Container for segmentation results."""

    original_coord: np.ndarray  # Nx3 array of original point coordinates
    labels: np.ndarray  # N array of classification labels


class PointCloudSource(ABC):
    """Abstract base class for point cloud data sources."""

    @abstractmethod
    def load(self, features: list[Feature]) -> PointCloudData:
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

    def load(self, features: list[Feature]) -> PointCloudData:
        """Load point cloud from file."""
        suffix = self.filepath.suffix.lower()

        if suffix == ".ply":
            return self._load_ply(features)
        elif suffix in (".las", ".laz"):
            return self._load_las(features)
        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. Supported: .ply, .las, .laz")

    def _load_ply(self, features: list[Feature]) -> PointCloudData:
        """Load PLY file."""

        with open(self.filepath, "rb") as f:
            cloud = PlyData.read(f)

        vertices = cloud["vertex"]
        xyz = np.vstack((vertices["x"], vertices["y"], vertices["z"])).T

        features_array: np.ndarray | None = None
        feature_list = []

        for feat in features:
            feat_name = "scalar_" + feat.name
            if feat_name not in vertices:
                raise ValueError(f"PLY file missing required scalar fields: {feat_name}")
            feat_data = vertices[feat_name]
            if not np.all(np.isfinite(feat_data)):
                raise ValueError(f"Inf values detected in scalar fields ({feat_name})")
            feature_list.append(feat.normalize(feat_data))

        if feature_list:
            features_array = np.column_stack(feature_list)

        return PointCloudData(xyz=xyz, features=features_array, source_name=self.get_name())

    def _load_las(self, features: list[Feature]) -> PointCloudData:
        """Load LAS/LAZ file."""

        las = laspy.read(self.filepath)
        xyz = np.vstack((las.x, las.y, las.z)).T

        features_array: np.ndarray | None = None
        feature_list = []

        for feat in features:
            if not hasattr(las, feat.name):
                raise ValueError(f"LAS file missing required scalar fields: {feat.name}")
            feat_data = getattr(las, feat.name)
            if not np.all(np.isfinite(feat_data)):
                raise ValueError(f"Inf values detected in scalar fields ({feat.name})")
            feature_list.append(feat.normalize(feat_data))

        if feature_list:
            features_array = np.column_stack(feature_list)

        return PointCloudData(xyz, features_array, self.get_name())


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
