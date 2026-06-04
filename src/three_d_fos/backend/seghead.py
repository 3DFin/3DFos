import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from three_d_fos.backend.liteptv1m1_model import LitePT
from three_d_fos.backend.ptv3v1m1_model import PointTransformerV3
from three_d_fos.backend.structure import Point

logger = logging.getLogger(__name__)

# Model constants
NUM_CLASSES = 4
PTV3_OUT_CHANNELS = 64
LITEPT_OUT_CHANNELS = 72


class SegmentationHeadV2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()
        self.backbone = backbone

    def forward(self, input_dict: dict) -> dict:
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat

        seg_logits = self.seg_head(feat)
        return dict(seg_logits=seg_logits)


def try_load_model(ckpt_path: Path | None = None, backbone: str = "ptv3") -> dict[str, Any]:
    """Load model checkpoint from local path or download from Github releases (torch.hub mecanism)."""
    if ckpt_path and ckpt_path.is_file():
        logger.info("Loading checkpoint from local path: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        model_url = f"https://github.com/3DFin/3DFos/releases/download/v0.0.1/{backbone}_3dfos_005.pth"
        logger.info("Using torch.hub model at %s", model_url)
        ckpt = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    return ckpt


def load(
    ckpt_path: Path | None = None,
    custom_config: dict | None = None,
    backbone: str = "ptv3",
) -> nn.Module:
    backbone_lower = backbone.lower()
    if backbone_lower == "ptv3":
        model = SegmentationHeadV2(
            num_classes=NUM_CLASSES,
            backbone_out_channels=PTV3_OUT_CHANNELS,
            backbone=PointTransformerV3(**custom_config),
        )
    elif backbone_lower == "litept":
        model = SegmentationHeadV2(
            num_classes=NUM_CLASSES,
            backbone_out_channels=LITEPT_OUT_CHANNELS,
            backbone=LitePT(**custom_config),
        )
    else:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Choose from: ptv3, litept")

    ckpt = try_load_model(ckpt_path, backbone_lower)

    torchsparse_statedict: dict[str, Any] = {}

    # State dict remapping for torchsparse++ / nanoTSparse
    # Both PTV3 and LitePT use sparse convolutions that need weight remapping
    # Weights have different names and shape, bias is ok.
    if backbone_lower in ("ptv3", "litept"):
        logger.info("%s state dict remapping for Torchsparse++ / [nano]TS", backbone.upper())
        for k, v in ckpt["state_dict"].items():
            if "cpe.0.weight" in k or "conv.weight" in k or "conv.0.weight" in k:
                v = v.permute(3, 2, 1, 4, 0)
                v = v.reshape(-1, v.shape[3], v.shape[4])
                k = k.replace("weight", "kernel")
            torchsparse_statedict[k] = v
        model.load_state_dict(torchsparse_statedict)
    else:
        model.load_state_dict(ckpt["state_dict"])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_parameters / 1e6)
    return model
