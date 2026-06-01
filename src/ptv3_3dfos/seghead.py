import torch
import torch.nn as nn

from ptv3_3dfos.ptv3v1m1_model import PointTransformerV3
from ptv3_3dfos.liteptv1m1_model import LitePT
from ptv3_3dfos.structure import Point


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


def load(
    ckpt_path: str = None,
    custom_config: dict = None,
    backbone: str = "ptv3",
):
    import os

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint in local path: {ckpt_path} ...")
        from packaging import version

        if version.parse(torch.__version__) >= version.parse("2.4"):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        model_url = "https://github.com/3DFin/PTV3_3DFos/releases/download/v0.0.1/ptv3_3dfos_005.pth"
        print(f"Using torch.hub model at {model_url}")
        ckpt = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")

    if backbone.lower() == "ptv3":
        model = SegmentationHeadV2(
            num_classes=4,
            backbone_out_channels=64,
            backbone=PointTransformerV3(**custom_config),
        )
    elif backbone.lower() == "litept":
        model = SegmentationHeadV2(
            num_classes=4,
            backbone_out_channels=72,
            backbone=LitePT(**custom_config),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'ptv3' or 'litept'")

    torchsparse_statedict = {}

    # Pointcept state dict remapping for torchsparse++ / nanoTSparse.
    # Weights have different names and and shape
    # Bias are ok
    if backbone.lower() == "ptv3" or backbone.lower() == "litept":
        print("PTV3 state dict remapping for Torchsparse++ / [nano]TS")
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
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
