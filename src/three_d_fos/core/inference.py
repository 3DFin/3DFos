import logging
import time

import numpy as np
import pgeof
import torch
from dendroptimized import voxelize
from nanotsparse.nn import functional as F

import three_d_fos
from three_d_fos.backend.tiling import RecursiveMainXYAxisTilingMask
from three_d_fos.io import PointCloudData

logger = logging.getLogger(__name__)


def preprocess(
    data: PointCloudData,
    grid_size: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Voxelize and compute normals."""
    xyz = data.xyz.astype(np.float64)
    voxelated_cloud, remap_ids, sample_ids = voxelize(xyz, grid_size, grid_size, 5, with_n_points=False, verbose=False)

    # Resample the data according to the voxelization.
    global_shift = np.min(xyz, axis=0)
    grid_coords = np.floor((voxelated_cloud - global_shift) / grid_size).astype(np.int_)

    remap_ids = remap_ids.astype(np.uint32)
    xyz_sampled = xyz[sample_ids]

    normals = pgeof.compute_features_selected(
        xyz_sampled,
        search_radius=0.5,
        max_knn=10000,
        selected_features=[
            pgeof.EFeatureID.Normal_x,
            pgeof.EFeatureID.Normal_y,
            pgeof.EFeatureID.Normal_z,
        ],
    )

    normals = normals.astype(np.float32)
    if data.features is not None:
        features = np.concatenate([normals, data.features[sample_ids].astype(np.float32)], axis=1)
    else:
        features = normals

    data_dict = {
        "grid_size": grid_size,
        "grid_coord": grid_coords,
        "coord": (xyz_sampled - global_shift).astype(np.float32),  # shift cloud to avoid quantization / stabilize
        "feat": features,
    }

    return data_dict, remap_ids, sample_ids


def run_inference(model: torch.nn.Module, data: dict, device: torch.device, tiling_factor: int) -> np.ndarray:
    """Perform tiled inference if needed and return raw classification labels."""

    transform = three_d_fos.transform.transform_config()

    if tiling_factor > 0:
        num_base_points = data["coord"].shape[0]
        all_labels = np.zeros(num_base_points, dtype=np.int64)

        tiler = RecursiveMainXYAxisTilingMask(tiling_factor)
        tile_mask = tiler.tile(data["coord"])
        unique_tile_ids = np.unique(tile_mask)

        for tile_id in unique_tile_ids:
            tile_data = {
                "grid_size": data["grid_size"],
                "grid_coord": data["grid_coord"][tile_mask == tile_id],
                "coord": data["coord"][tile_mask == tile_id],
                "feat": data["feat"][tile_mask == tile_id],
            }

            transformed_data = transform(tile_data)
            start_infer_tile = time.time()
            logger.info("Running inference on tile %i/%i...", tile_id, len(unique_tile_ids))
            tile_mask_indicator = tile_mask == tile_id

            # Run inference on this tile
            tile_labels = run_inference_one_tile(model, transformed_data, device)

            # Store labels at the correct remap_ids positions
            all_labels[tile_mask_indicator] = tile_labels
            logger.info("Inference on tile done in %.2f seconds.", time.time() - start_infer_tile)
        # remap the labels to the original PC
        return all_labels
    else:
        transformed_data = transform(data)
        return run_inference_one_tile(model, transformed_data, device)


def run_inference_one_tile(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
) -> np.ndarray:
    """Perform inference and return raw classification labels."""

    with torch.inference_mode():
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, non_blocking=True)

        # Configure nanots according to our current device.
        nanots_config = F.conv_config.get_default_conv_config()

        if device.type == "cuda":
            nanots_config.dataflow = F.Dataflow.ImplicitGEMM
        else:
            nanots_config.dataflow = F.Dataflow.GatherScatter
            nanots_config.kmap_mode = "hashmap"

        F.conv_config.set_global_conv_config(nanots_config)

        predictions = model(data)
        return predictions["seg_logits"].argmax(dim=-1).cpu().numpy()
