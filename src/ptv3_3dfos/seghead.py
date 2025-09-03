import torch
import torch.nn as nn
from .structure import Point
from .oacnns_model import OACNNs
from .ptv3_model import PointTransformerV3

class SegmentationHeadV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = backbone

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat

        seg_logits = self.seg_head(feat)
        return dict(seg_logits=seg_logits)


class SegmentationHeadV1(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        return dict(seg_logits=seg_logits)


def load(
    name: str = "3dfos",
    custom_config: dict = None,
    backbone: str = "oacnns",
):
    import os
    if os.path.isfile(name):
        print(f"Loading checkpoint in local path: {name} ...")
        ckpt_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    from packaging import version
    if version.parse(torch.__version__) >= version.parse("2.4"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if backbone.lower() == "ptv3":
        model = SegmentationHeadV2(num_classes=4, backbone_out_channels=64, backbone=PointTransformerV3(**custom_config))
    elif backbone.lower() == "oacnns":
        model = SegmentationHeadV1(backbone=OACNNs(**custom_config))
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'ptv3' or 'oacnns'")

    model.load_state_dict(ckpt["state_dict"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
