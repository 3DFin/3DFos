import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from three_d_fos.backend.model.ptv3v1m1_model import PointTransformerV3
from three_d_fos.backend.structure import Point
from three_d_fos.core.model import PTV3_FULL_MODEL, Model

logger = logging.getLogger(__name__)

# Model constants
NUM_CLASSES = 4


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


def try_load_model(ckpt_path: Path | None, url: str | None) -> dict[str, Any]:
    """Load model checkpoint from local path or download from Github releases (torch.hub mechanism)."""
    if ckpt_path and ckpt_path.is_file():
        logger.info("Loading checkpoint from local path: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    elif url:
        logger.info("Using torch.hub model at %s", url)
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    else:
        raise RuntimeError("Unable to load model ()")
    return ckpt


def load(
    ckpt_path: Path | None = None,
    backbone_model: Model = PTV3_FULL_MODEL,
) -> nn.Module:
    model = SegmentationHeadV2(
        num_classes=NUM_CLASSES,
        backbone_out_channels=backbone_model.output_features,
        backbone=backbone_model.model_instance,
    )

    ckpt = try_load_model(ckpt_path, backbone_model.url)

    torchsparse_statedict: dict[str, Any] = {}

    # State dict remapping for torchsparse++ / nanoTSparse
    # Both PTV3 and LitePT use sparse convolutions that need weight remapping
    # Weights have different names and shape, bias is ok.
    logger.info("%s state dict remapping for Torchsparse++ / [nano]TS", backbone_model.name)
    for k, v in ckpt["state_dict"].items():
        if "cpe.0.weight" in k or "conv.weight" in k or "conv.0.weight" in k:
            v = v.permute(3, 2, 1, 4, 0)
            v = v.reshape(-1, v.shape[3], v.shape[4])
            k = k.replace("weight", "kernel")
        torchsparse_statedict[k] = v
    model.load_state_dict(torchsparse_statedict)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_parameters / 1e6)
    return model
