"""
General utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.backends import cudnn

logger = logging.getLogger(__name__)


def is_gpu_ampere_or_newer():
    if not torch.cuda.is_available():
        return False

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()
    cc = float(f"{major}.{minor}")
    logger.info("Compute Capability  %.2f", cc)

    # Ampere = 8.x / TODO maybe we need to target 8.6
    if cc >= 8.0:
        logger.info("GPU is Ampere or newer")
        return True
    else:
        logger.info("GPU older than Ampere")
        return False


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def get_random_seed():
    seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
