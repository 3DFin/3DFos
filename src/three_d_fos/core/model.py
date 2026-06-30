import logging
import sys
from dataclasses import dataclass, field
from functools import reduce

import torch

from three_d_fos.backend.model.liteptv1m1_model import LitePT
from three_d_fos.backend.model.ptv3v1m1_model import PointTransformerV3
from three_d_fos.backend.module import PointModule
from three_d_fos.core.feature import DIST_AXES, Z0, Feature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    backbone: type[PointModule]
    url: str | None
    output_features: int
    config: dict = field(default_factory=dict)
    additional_features: set[Feature] = field(default_factory=set)

    @property
    def features(self) -> frozenset[Feature]:
        return frozenset(self.additional_features)

    @property
    def feature_size(self) -> int:
        return 3 + reduce(lambda size, f: size + f.size, self.features, 0)

    @property
    def model_config(self) -> dict:
        self.config["in_channels"] = self.feature_size
        return self.config

    @property
    def model_instance(self) -> PointModule:
        return self.backbone(**self.model_config)


PTV3_OUT_CHANNELS = 64
LITEPT_OUT_CHANNELS = 72


def do_support_flash_attn():
    # Flash attn backend is not available on windows builds of PyTorch.
    if not torch.cuda.is_available() or sys.platform == "win32":
        return False

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()
    cc = float(f"{major}.{minor}")
    logger.info("Compute Capability  %.2f", cc)

    # Ampere = 8.x / TODO maybe we need to target 8.6
    if cc >= 8.0:
        logger.info("GPU is Ampere or newer - Flash attention enabled")
        return True
    else:
        logger.info("GPU older than Ampere - No Flash attention")
        return False


BASE_LITEPT_CONFIG: dict = dict(
    in_channels=5,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
    dec_depths=(0, 0, 0, 0),
    dec_channels=(72, 72, 144, 252),
    dec_num_head=(4, 4, 8, 14),
    dec_patch_size=(1024, 1024, 1024, 1024),
    dec_conv=(False, False, False, False),
    dec_attn=(False, False, False, False),
    dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    pre_norm=True,
    shuffle_orders=True,
    enc_mode=False,
    enable_flash=do_support_flash_attn(),
)

BASE_PTV3_CONFIG: dict = dict(
    in_channels=5,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,  # no-op in eval
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=do_support_flash_attn(),
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)

# Predefined models
PTV3_FULL_MODEL = ModelDefinition(
    name="PTv3_full",
    backbone=PointTransformerV3,
    url="https://github.com/3DFin/3DFos/releases/download/v0.1.0/ptv3_3dfos_005.pth",
    output_features=PTV3_OUT_CHANNELS,
    additional_features={Z0, DIST_AXES},
    config=BASE_PTV3_CONFIG,
)

LITEPT_FULL_MODEL = ModelDefinition(
    name="LitePT_full",
    backbone=LitePT,
    url="https://github.com/3DFin/3DFos/releases/download/v0.1.0/litept_3dfos_005.pth",
    output_features=LITEPT_OUT_CHANNELS,
    additional_features={Z0, DIST_AXES},
    config=BASE_LITEPT_CONFIG,
)

LITEPT_NORMALS_Z0_MODEL = ModelDefinition(
    name="LitePT_normals_z0",
    backbone=LitePT,
    url="https://github.com/3DFin/3DFos/releases/download/v0.2.0/litept_3dfos_normals_z0_005.pth",
    output_features=LITEPT_OUT_CHANNELS,
    additional_features={Z0},
    config=BASE_LITEPT_CONFIG,
)

LITEPT_NORMALS_MODEL = ModelDefinition(
    name="LitePT_normals",
    backbone=LitePT,
    url="https://github.com/3DFin/3DFos/releases/download/v0.2.0/litept_3dfos_normals_005.pth",
    output_features=LITEPT_OUT_CHANNELS,
    config=BASE_LITEPT_CONFIG,
)

# Model Map
MODEL_MAP: dict[str, ModelDefinition] = {
    "ptv3_full": PTV3_FULL_MODEL,
    "litept_full": LITEPT_FULL_MODEL,
    "litept_normals_z0": LITEPT_NORMALS_Z0_MODEL,
    "litept_normals": LITEPT_NORMALS_MODEL,
}
