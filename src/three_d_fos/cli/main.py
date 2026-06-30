import argparse
import logging
import time
from pathlib import Path

import torch

import three_d_fos
from three_d_fos.cli.logging import setup_logging
from three_d_fos.core import inference as backend_inference
from three_d_fos.core.model import MODEL_MAP
from three_d_fos.io import FilePointCloudDestination, FilePointCloudSource, SegmentationResult

logger = logging.getLogger(__name__)
setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Point Cloud Segmentation for forestry applications.")
    parser.add_argument("input_path", type=Path, help="Path to the input PLY or LAS file")
    parser.add_argument("--model_path", type=Path, help="Path to the model file (.pth)")
    parser.add_argument(
        "--output_path",
        type=Path,
        default="seg_result.las",
        help="Output LAS file path",
    )
    parser.add_argument("--grid_size", type=float, default=0.1, help="Voxel grid size")
    parser.add_argument(
        "--tiling_factor", type=int, default=0, help="Number of tiles. Total number of tiles is 2^tiling_factor"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_MAP.keys()),
        default="ptv3_full",
        help=f"Choose model: {','.join(MODEL_MAP.keys())} ",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Choose device that host the computation",
    )

    args = parser.parse_args()

    # Check CUDA availability and set device
    device: torch.device = torch.device(args.device)

    if device.type == "cuda":
        logger.info("Using CUDA: %s", torch.cuda.get_device_name())
    else:
        logger.info("Using CPU")

    start_total = time.time()

    start_model = time.time()
    model_definition = MODEL_MAP[args.model]
    model = three_d_fos.seghead.load(ckpt_path=args.model_path, model_definition=model_definition)

    model.to(device).eval()
    logger.info("Model loaded in %.2f seconds.", time.time() - start_model)

    start_data = time.time()
    logger.info("Loading data from %s...", args.input_path)

    source = FilePointCloudSource(args.input_path, model_definition.features)
    data = source.load()
    original_coord = data.xyz.copy()

    logger.info("Data loaded in %.2f seconds.", time.time() - start_data)

    start_preproc = time.time()
    logger.info("Running Preprocessing...")
    point_features, remap_ids, _ = backend_inference.preprocess(data, args.grid_size)
    logger.info("Preprocessing done in %.2f seconds.", time.time() - start_preproc)

    start_infer = time.time()
    logger.info("Running inference...")
    labels = backend_inference.run_inference(model, point_features, device, args.tiling_factor)[remap_ids]
    logger.info("Full inference done in %.2f seconds.", time.time() - start_infer)

    destination = FilePointCloudDestination(args.output_path)
    result = SegmentationResult(original_coord=original_coord, labels=labels)
    destination.save(result)

    logger.info("Total time: %.2f seconds.", time.time() - start_total)


if __name__ == "__main__":
    main()
