"""
Backend inference module

Authors: Romain Janvier, Diego Laíño Rebollido, Carlos Cabo, Diego
Copyright: Department of Mining Exploitation of the University of Oviedo (Spain)
"""

import numpy as np
import pgeof
import torch
from dendroptimized import voxelize

# Feature normalization constants
DIST_AXES_SCALE = 15.0
Z0_SCALE = 30.0


def normalize_scalar_fields(dist_axes: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize 3DFin scalar fields to [0, 1] range."""
    dist_axes = np.clip(dist_axes / DIST_AXES_SCALE, 0.0, 1.0)
    z0 = np.clip(z0 / Z0_SCALE, 0.0, 1.0)
    return dist_axes, z0


def preprocess(
    xyz: np.ndarray, z0: np.ndarray, dist_axes: np.ndarray, grid_size: float
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Voxelize and compute normals."""
    xyz = xyz.astype(np.float64)
    voxelated_cloud, remap_ids, sample_ids = voxelize(xyz, grid_size, grid_size, 5, with_n_points=False, verbose=False)

    # Resample the data according to the voxelization
    global_shift = np.min(xyz, axis=0)
    grid_coords = np.floor((voxelated_cloud - global_shift) / grid_size).astype(np.int_)

    remap_ids = remap_ids.astype(np.uint32)
    xyz_sampled = xyz[sample_ids]

    normals = pgeof.compute_features_selected(
        xyz_sampled,
        search_radius=0.5,
        max_knn=500000,
        selected_features=[
            pgeof.EFeatureID.Normal_x,
            pgeof.EFeatureID.Normal_y,
            pgeof.EFeatureID.Normal_z,
        ],
    )

    features = {
        "grid_size": grid_size,
        "grid_coord": grid_coords,
        "coord": (xyz_sampled - global_shift).astype(np.float32),  # shift cloud to avoid quantization / stabilize
        "normal": normals.astype(np.float32),
        "z0": np.expand_dims(z0[sample_ids], axis=1).astype(np.float32),
        "dist_axes": np.expand_dims(dist_axes[sample_ids], axis=1).astype(np.float32),
    }

    return features, remap_ids, sample_ids


def run_inference(
    model: torch.nn.Module,
    data: dict,
    remap_ids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Perform inference and return raw classification labels."""
    with torch.inference_mode():
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, non_blocking=True)

        predictions = model(data)

        return predictions["seg_logits"][remap_ids].argmax(dim=-1).cpu().numpy()
