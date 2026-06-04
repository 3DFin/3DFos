import argparse
import logging
import sys
import time
from pathlib import Path

import torch

import three_d_fos
from three_d_fos.backend import inference as backend_inference
from three_d_fos.cli import io
from three_d_fos.cli.logging import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

try:
    import flash_attn
except ImportError:
    logger.warning("no flash attn")
    flash_attn = None


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Point Cloud Segmentation for forestry applications.")
    parser.add_argument("input_path", type=Path, help="Path to the input PLY file")
    parser.add_argument("--model_path", type=Path, help="Path to the model file (.pth)")
    parser.add_argument(
        "--output_path",
        type=Path,
        default="seg_result.las",
        help="Output LAS file path",
    )
    parser.add_argument("--grid_size", type=float, default=0.1, help="Voxel grid size")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["ptv3", "litept"],
        default="ptv3",
        help="Choose backbone: ptv3 or litept",
    )

    args = parser.parse_args()

    # Check CUDA availability and set device
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        logger.info("Using CUDA: %s", torch.cuda.get_device_name())
    else:
        logger.warning("CUDA not available, using CPU")

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
    logger.info("Model loaded in %.2f seconds.", time.time() - start_model)

    start_data = time.time()
    suffix = args.input_path.suffix.lower()

    if suffix == ".ply":
        xyz, z0, dist_axes = io.read_ply(args.input_path)
    elif suffix in (".las", ".laz"):
        xyz, z0, dist_axes = io.read_las(args.input_path)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Supported: .ply, .las, .laz")

    dist_axes, z0 = backend_inference.normalize_scalar_fields(dist_axes, z0)
    original_coord = xyz.copy()

    logger.info("Data loaded in %.2f seconds.", time.time() - start_data)

    start_preproc = time.time()
    logger.info("Running Preprocessing...")
    point_features, remap_ids, _ = backend_inference.preprocess(xyz, z0, dist_axes, args.grid_size)
    transformed_point = transform(point_features)
    logger.info("Preprocessing done in %.2f seconds.", time.time() - start_preproc)

    start_infer = time.time()
    logger.info("Running inference...")
    labels = backend_inference.run_inference(model, transformed_point, remap_ids, device)
    io.write_las_predictions(str(args.output_path), original_coord, labels)
    logger.info("Inference done in %.2f seconds.", time.time() - start_infer)

    logger.info("Total time: %.2f seconds.", time.time() - start_total)


if __name__ == "__main__":
    main()
