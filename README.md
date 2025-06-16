# PTV3-3DFos

Minimal PTV33 standalone inspired by [sonata](https://github.com/facebookresearch/sonata) standalone.

## Changes:
- Removed torch_scatter dependencies (replaced by pure torch one from PYG).
- Changed sonata model for original PTV3 one.
- Added a dedicated inference demo/script for 3DFos datasets

## Installation:

```
uv sync --extra cu126
```

Flash attention lower the memory usage and improve the runtime but it's not mandatory, 
if you want flash-attn, you have to run this command **AFTER** the first one.

```
uv sync --extra cu126 --extra flash-attn
```

Flash attention is only compatible with NVIDIA cards have a compute capability of 80+. 
You might need the CUDA compiler, which is part of the CUDA toolkit, in order to compile flash-attn. 
This could be be very time-consuming, particularly on Windows (On Linux, the install script will attempt to download pre-compiled binary wheel from github)

## usage

```
uv run 3DFos path_to_the_model.pth path_to_the_cloud.las [--grid_size 0.05]
```

The point Cloud could be in las/laz or ply format. 


## TODO:
- Handle global shift for las clouds
- Retest CPU only scenario and find what is wrong with spconv. spconv is not even compiled for CPU on windows. 
