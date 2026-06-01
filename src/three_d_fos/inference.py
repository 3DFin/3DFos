"""
Main inference script

Authors: Romain Janvier, Diego Laíño Rebollido, Carlos Cabo, Diego
Copyright: Department of Mining Exploitation of the University of Oviedo (Spain)
"""

import argparse
import time
from pathlib import Path

import laspy
import numpy as np
import torch
from plyfile import PlyData

import pgeof
import three_d_fos
import three_d_fos.seghead
import three_d_fos.liteptv1m1_model
from dendroptimized import voxelize

try:
    import flash_attn
except ImportError:
    print("no flash attn")
    flash_attn = None

# Feature normalization constants
DIST_AXES_SCALE = 15.0
Z0_SCALE = 30.0


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
    voxelated_cloud, remap_ids, sample_ids = voxelize(
        xyz, grid_size, grid_size, 5, with_n_points=False, verbose=False
    )

    # Resample the data according to the voxelization
    global_shift = np.min(xyz, axis=0)
    grid_coords = np.floor((voxelated_cloud - global_shift) / grid_size).astype(np.int_)

    remap_ids = remap_ids.astype(np.uint32)
    xyz_sampled = xyz[sample_ids]

    normals = pgeof.compute_features_selected(
        xyz_sampled,
        search_radius=0.5,  # TODO: could be exposed as a parameter
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
    original_coord: np.ndarray,
    output_path: Path,
    device: torch.device,
) -> None:
    """Perform inference and save predictions as LAS."""
    with torch.inference_mode():
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, non_blocking=True)

        predictions = model(data)

        labels = predictions["seg_logits"][remap_ids].argmax(dim=-1).cpu().numpy()

        header = laspy.LasHeader(version="1.4", point_format=6)
        las = laspy.LasData(header)
        las.x, las.y, las.z = original_coord.T
        las.classification = labels
        las.write(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Point Cloud Segmentation for forestry applications.")
    parser.add_argument("input_path", type=Path, help="Path to the input PLY file")
    parser.add_argument("--model_path", type=Path, help="Path to the model file (.pth)")
    parser.add_argument("--output_path", type=Path, default="seg_result.las", help="Output LAS file path")
    parser.add_argument("--grid_size", type=float, default=0.1, help="Voxel grid size")
    parser.add_argument("--backbone", type=str, choices=["ptv3", "litept"], default="ptv3", help="Choose backbone: ptv3 or litept")

    args = parser.parse_args()

    # Check CUDA availability and set device
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    start_total = time.time()

    start_model = time.time()
    transform = three_d_fos.transform.transform_config()
    if args.backbone == "ptv3":
        config = three_d_fos.ptv3v1m1_model.model_config()
        config["enable_flash"] = bool(flash_attn)
        model = three_d_fos.seghead.load(ckpt_path=args.model_path, custom_config=config, backbone="ptv3")
    elif args.backbone == "litept":
        config = three_d_fos.liteptv1m1_model.model_config()
        config["enable_flash"] = bool(flash_attn)
        model = three_d_fos.seghead.load(ckpt_path=args.model_path, custom_config=config, backbone="litept")
    else:
        raise ValueError(f"Unsupported backbone: '{args.backbone}'. Choose from: ptv3, litept")

    model.to(device).eval()
    print(f"Model loaded in {time.time() - start_model:.2f} seconds.")

    start_data = time.time()
    suffix = args.input_path.suffix.lower()

    if suffix == ".ply":
        xyz, z0, dist_axes = read_ply(args.input_path)
    elif suffix in (".las", ".laz"):
        xyz, z0, dist_axes = read_las(args.input_path)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Supported: .ply, .las, .laz")

    dist_axes, z0 = normalize_scalar_fields(dist_axes, z0)
    original_coord = xyz.copy()

    print(f"Data loaded in {time.time() - start_data:.2f} seconds.")

    start_preproc = time.time()
    print("Running Preprocessing...")
    point_features, remap_ids, _ = preprocess(xyz, z0, dist_axes, args.grid_size)
    transformed_point = transform(point_features)
    print(f"Preprocessing done in {time.time() - start_preproc:.2f} seconds.")

    start_infer = time.time()
    print("Running inference...")
    run_inference(model, transformed_point, remap_ids, original_coord, args.output_path, device)
    print(f"Inference done in {time.time() - start_infer:.2f} seconds.")

    print(f"Total time: {time.time() - start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
