# PTV3-3DFos

Minimal PTV3 standalone inspired by [sonata](https://github.com/facebookresearch/sonata) standalone.

## Changes:

- Added clean uv packaging
- Removed torch_scatter dependencies (replaced scatter one from PYG by pure torch calls, simplify dependencies).
- [WIP] Replaced spconv by torchsparse++ / nanoTS for sparse convolution. nanoTS is not affected by CUMM bugs (like https://github.com/FindDefinition/cumm/issues/26) and is easier
  to package / maintain
- Use toch built-in SDPA to levrage efficient and memory friendly attention kernels
- Added a dedicated inference demo/script for 3DFos datasets

## Installation:

```
uv sync --extra cu126
```

then

```
uv sync --extra cu130 --extra nanots
```

Flash attention lowers memory usage and improves runtime, but it's not mandatory.
if you want flash-attn, you have to run this command **AFTER** the first one.

```
uv sync --extra cu130 --extra nanots --extra flash-attn
```

Flash attention is only compatible with NVIDIA cards that have a compute capability of 8.0+.
You might need the CUDA compiler, which is part of the CUDA toolkit, in order to compile flash-attn.
This could be very time-consuming, particularly on Windows. (On Linux, the install script will attempt to download a pre-compiled binary wheel from GitHub.)

## usage

```
uv run 3DFos <path_to_the_model.pth> <path_to_the_cloud.las|ply> [--grid_size 0.05] [--backbone ptv3]
```

The point cloud can be in las/laz or ply format.

## TODO:

- Handle global shift for LAS clouds?
- Use point closest to the voxel center.
- Add a spatial tiling mechanism?
- Add LightPT model.
- Retest CPU only scenario and find what is wrong with spconv => spconv is not even compiled for CPU on windows
- Citation section with PTV3/PointCept references (and sonata?)
- Where to store the weights?
