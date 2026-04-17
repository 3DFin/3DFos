# PTV3-3DFos

Minimal PTV3 standalone for forestry application, inspired by [sonata](https://github.com/facebookresearch/sonata) standalone.

Please cite [PointCept](https://github.com/pointcept/pointcept), PTV3 and Sonata if you use this work (see PointCept details for details).

PTV3-3DFos has been developed at the Centre of Wildfire Research of Swansea University (UK) in collaboration with the Research Institute of Biodiversity (CSIC, Spain) and the Department of Mining Exploitation of the University of Oviedo (Spain).

Funding provided by the UK NERC project (NE/T001194/1):

'Advancing 3D Fuel Mapping for Wildfire Behaviour and Risk Mitigation Modelling'

and by the Spanish Knowledge Generation project (PID2021-126790NB-I00):

‘Advancing carbon emission estimations from wildfires applying artificial intelligence to 3D terrestrial point clouds’.

## Changes vs sonata:

- Added clean uv packaging
- Removed torch_scatter dependencies (replaced scatter one from PYG by pure torch calls, simplify dependencies).
- Replaced spconv by Torchsparse++ / [nano]TSparse for sparse convolution. [nano]TSparse is not affected by CUMM bugs (like https://github.com/FindDefinition/cumm/issues/26) and is easier
  to package / maintain. We do not provide yet pre compiled version of [nano]TSparse, you may need a C/C++ and CUDA compiler in order to run this code.
- Use torch built-in SDPA to leverage efficient and memory friendly attention kernels. This removes the need for flash attention.
- Added a dedicated inference demo/script for 3DFos datasets

## Installation:

```
uv sync --extra cu130
```

then

```
uv sync --extra cu130 --extra nanotsparse
```

Flash attention lowers memory usage and improves runtime, but it's not mandatory.
If you want to use `flash-attn` package, you have to run this command **AFTER** the first one.

```
uv sync --extra cu130 --extra nanotsparse --extra flash-attn
```

Flash attention is only compatible with NVIDIA cards that have a compute capability of 8.0+.
You might need the CUDA compiler, which is part of the CUDA toolkit, in order to compile flash-attn.
This could be very time-consuming, particularly on Windows. (On Linux, the install script will attempt to download a pre-compiled binary wheel from GitHub.)

## Usage

```
uv run 3DFos <path_to_the_model.pth> <path_to_the_cloud.las|ply> [--grid_size 0.05] [--backbone ptv3]
```

Point clouds can be in las/laz or ply format.

## TODO:

- Use point closest to the voxel center.
- Add a spatial tiling mechanism?
- Add LightPT model.
- Use torch varlen.
- Where to store the weights?
