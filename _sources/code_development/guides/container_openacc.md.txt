---
orphan: true
---

# Container with GPU support (OpenACC)

```{note}
To follow this tutorial you need to have root access to a Linux computer
with Singularity installed, e.g. your personal laptop/workstation.
Please follow the installation
[instructions](https://sylabs.io/guides/3.7/user-guide/quick_start.html)
from the Singularity documentation.
```

This example demonstrates:
1. how to build a (Nvidia) GPU container using a Nvidia GPU Cloud (NGC) base image
2. how to copy a file into the container
3. how to compile a simple OpenACC program inside the container
4. how to run the container on a cluster with GPU resourses

This example is based on the {ref}`OpenACC tutorial <openacc>`, and we will simply copy
the fully optimized source code for the Jacobi iteration to serve as our GPU application:

```{eval-rst}
:download:`jacobi_optimized.c <./openacc/jacobi_optimized.c>`
```

**Writing the definition file**

For this example we will use a base image from the
[Nvidia GPU Cloud](https://ngc.nvidia.com/),
which includes everything we need for this simple application. The NGC hosts a
wide variety of different Nvidia based container images, supporting different
CUDA versions and operating systems. We will choose the
[NVIDIA HPC SDK](https://ngc.nvidia.com/catalog/containers/nvidia:nvhpc)
container since we need the `nvc` compiler for OpenACC (the standard NVIDIA CUDA
image does not have this). We select the latest development package on Ubuntu-20.04:
```
Bootstrap: docker
From: nvcr.io/nvidia/nvhpc:21.5-devel-cuda11.3-ubuntu20.04

%post
    apt-get update

%files
    jacobi_optimized.c /work/jacobi_optimized.c

%post
    nvc -g -fast -acc -Minfo=accel -o /usr/local/bin/jacobi /work/jacobi_optimized.c
```

Here we assume that we have downloaded the source file `jacobi_optimized.c` from the link above
and put it in the same directory as the definition file. We then copy the source file into the
container with the `%files` section, and then compile it with `nvc` using the same string of options
as in the original tutorial while putting the output executable into a runtime path (`/usr/local/bin`).

```{tip}
We need the `devel` base image in order to compile the application inside the container,
but once it's built we can get away with a `runtime` base container for running it.
Check the official documentation on how to do
[multi-stage builds](https://sylabs.io/guides/3.7/user-guide/definition_files.html#multi-stage-builds),
which can significantly reduce the size of the final container image.
```

**Building the container**

We can now build the container with the following command (you need sudo rights for this step):
```console
[me@laptop]$ sudo singularity build example-openacc.sif example-openacc.def

[... lots of output ...]

INFO:    Creating SIF file...
INFO:    Build complete: example-openacc.sif
```

**Running the container**

Once `example-openacc.sif` is generated, we can `scp` the container file to the cluster:

```console
[me@laptop]$ scp example-openacc.sif me@saga.sigma2.no
```

We test the container in an interactive job on a GPU node:
```console
[me@login-1.SAGA]$ srun --ntasks=1 --gpus-per-task=1 --mem=2G --time=10:00 --partition=accel --account=<nnXXXXk> --pty bash
[me@c7-8.SAGA]$ time singularity exec --nv example-openacc.sif jacobi
real	0m3.748s
user	0m1.715s
sys     0m1.770s
```

Remember the `--nv` flag to expose the Nvidia GPU resources to the container.
Here we can see that the execution time is comparable to what is reported in the
original {ref}`tutorial <openacc>`.

