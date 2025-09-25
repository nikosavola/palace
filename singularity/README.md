<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
# Apptainer/Singularity Container

This directory contains a [definition file](singularity.def) for building *Palace*
containers using [Apptainer/Singularity](https://apptainer.org/).

## Quick start

Assuming you have installed Apptainer/Singularity and the `singularity` executable is on
your path, the container can be built with:

```bash
singularity build palace.sif singularity.def
```

and run with:

```bash
singularity run palace.sif <ARGS...>
```

where `<ARGS...>` is a list of command line arguments provided to the `palace` executable.

## GPU Support

The definition file supports optional GPU compilation through build arguments. To enable 
GPU support, use the `--build-arg` flag:

### CUDA Support

To build *Palace* with CUDA support for NVIDIA GPUs:

```bash
singularity build --build-arg PALACE_WITH_CUDA=ON palace.sif singularity.def
```

You may also want to enable GPU-aware MPI:

```bash
singularity build --build-arg PALACE_WITH_CUDA=ON --build-arg PALACE_WITH_GPU_AWARE_MPI=ON palace.sif singularity.def
```

### HIP Support

To build *Palace* with HIP support for AMD GPUs:

```bash
singularity build --build-arg PALACE_WITH_HIP=ON palace.sif singularity.def
```

### Running with GPU Support

When running the GPU-enabled container, use the appropriate Apptainer/Singularity GPU flags:

For NVIDIA GPUs:
```bash
singularity run --nv palace.sif <ARGS...>
```

For AMD GPUs:
```bash
singularity run --rocm palace.sif <ARGS...>
```

For more information about GPU support in Apptainer/Singularity, see the 
[GPU documentation](https://apptainer.org/docs/user/main/gpu.html).

For detailed instructions, see the documentation specific to
[building](https://awslabs.github.io/palace/dev/install/#Build-using-Singularity/Apptainer)
and [running](https://awslabs.github.io/palace/dev/run/#Singularity/Apptainer) *Palace*
with Apptainer/Singularity.
