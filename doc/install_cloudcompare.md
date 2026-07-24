CloudCompare Plugin Installation
=========================

Thanks to [CloudCompare-PythonRuntime](https://github.com/tmontaigu/CloudCompare-PythonRuntime) plugin, 3DFoS could be used inside CloudCompare as a plugin. However, it is not included by default in the CloudCompare distribution and requires additional user actions to install.

This document focuses on the **Windows** system using the **official CloudCompare distribution** and only supports `3DFoS` on cpu (no CUDA compatibility for now).
For macOS and Linux, or Windows+CUDA the process depends on the type of installation (e.g., `Flatpak`, bundle, or compied from source) and may require setting up a virtual environment using uv or pip. For a general overview, refer to the main [README](https://github.com/3DFin/3DFos). In this case, the only hard requirement is that the Python version of the virtual environment must be compatible and you might need a compiler toolchain installed on your computer.

## Windows installation

### Donwload CloudCompare

You need [CloudCompare 2.14.beta ](https://www.cloudcompare.org) at least. The first version known to be comptible with this install procedure is the beta from 07-23-2026.

### Create a dedicated virtual env.

By default CloudCompare come with a minimal Python distribution installed and sometimes you do not have to right to modify it, and in all case it's very risky and error prone. To handle this, `CloudCompare-PythonRuntime` provide the habitlity to create a self contained and isolated Python environment (called a `virtual environement` or `venv`) in order to install additional Python packages. we will take advantage of this to install 3DFoS plugin

### Install or upgrade the plugin

lauch `CloudCompare` and go to `plugins > Python plugin > show settings` menu

In the package name input path

```
3dfos[cpu, windowsplugin] @ git+https://github.com/3DFin/3DFos.git
```
