---
orphan: true
---

(sycl)=

# What is SYCL?

> [SYCL](https://www.khronos.org/sycl/) is a higher-level programming model to improve programming
> productivity on various hardware accelerators. It is a single-source domain-specific embedded
> language based on pure C++17. It is a standard developed by Khronos Group, announced in March 2014.
>
>>>>>>>>>>>>>>>>>>> ---[_Wikipedia_](https://en.wikipedia.org/wiki/SYCL)

(hipsycl)=
## What is hipSYCL?

[hipSYCL](https://github.com/illuhad/hipSYCL) is one of a few currently available implementations
of the SYCL standard (none of which is feature complete wrt SYCL-2020 at the time of writing).
hipSYCL provides backends for offloading to OpenMP (any type of CPU), CUDA (Nvidia GPUs) and
HIP/ROCm (AMD GPUs), as well as experimental support for Level Zero (Intel GPUs). This particular
SYCL implementation is interesting for us at NRIS because it provides a unified tool for all our
current hardware, which constitutes Intel and AMD CPUs, Nvidia GPUs on Saga and Betzy and AMD GPUs
on LUMI. Other available SYCL implementations are [Codeplay](https://developer.codeplay.com/home/)'s
ComputeCpp and [Intel oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html)'s
DPC++, which are currently more geared towards Intel hardware (CPUs, GPUs and FPGAs).

(hipsycl-start)=
# Getting started with hipSYCL

```{note}
In this tutorial we will use the global installation of hipSYCL on Saga.
If you want to use another SYCL implementation or you need to install it
on a different machine, please refer to the installation instructions in the
[SYCL Academy](https://github.com/codeplaysoftware/syclacademy) documentation.
```

## Hello world example

This example demonstrates:

1. how to compile a minimal SYCL application using the global hipSYCL installation on Saga
2. how to compile for different target architectures
3. how to run the example on a GPU node on Saga

In this example we will write a very simple program that queries the system for information
about which device it runs on. We will then compile the code for both CPU and (Nvidia) GPU
targets, and verify that it is able to find both devices. This can be achieved with
just a few lines of code:

```{eval-rst}
.. literalinclude:: hipsycl/hello_world.cpp
   :language: cpp
```

Here we first include the main SYCL header file, which in our case will be provided by the
`hipSYCL` module on Saga. In the main function we simply initialize a `sycl::queue` using the
so-called `default_selector`, and then we print out which device was picked up for this particular
queue (more on queues later). The `default_selector` will choose an accelerator if one is found
on the host, otherwise it will fall back to run as (traditional) OpenMP on the host CPU.
By specifying different types of `selectors` it is possible to e.g. force the code to always
run on the CPU, or to choose a particular device if more than one accelerator is available.

### Compiling for CPUs

In order to compile the code we need to have a SYCL implementation available, which we will
get by loading the `hipSYCL` module on Saga (check with `module avail hipsycl` to see which
versions are currently available):

```console
[me@login-1.SAGA ~]$ module load hipSYCL/0.9.1-gcccuda-2020b
```

```{note}
If you want to compile for Nvidia GPUs you need a `gcccuda` version of `hipSYCL`.
With EasyBuild there exists also a CPU-only version based on the `GCC` toolchain without
the CUDA backend (and hopefully soon a ROCm version for targeting AMD GPUs).
```

After loading the `hipSYCL` module you should have the `syclcc` compiler wrapper available
on your command line (try e.g. `syclcc --help`). We will first compile the code only for
CPU by specifying the `omp` target:

```console
[me@login-1.SAGA ~]$ syclcc --hipsycl-targets=omp -O3 -o hello_world hello_world.cpp
```

This step should hopefully pass without any errors or warnings. If we run the resulting
executable on the login node we will however see a warning:

```console
[me@login-1.SAGA ~]$ ./hello_world
[hipSYCL Warning] backend_loader: Could not load backend plugin: /cluster/software/hipSYCL/0.9.1-gcccuda-2020b/bin/../lib/hipSYCL/librt-backend-cuda.so
[hipSYCL Warning] libcuda.so.1: cannot open shared object file: No such file or directory
Chosen device: hipSYCL OpenMP host device
```

The reason for the warning is that we use a `hipSYCL` version that is compiled with the
CUDA backend, but we don't have the CUDA drivers available when we run the program on the
login nodes. But no worries, the last line that is printed is the actual output of our
program, which tells us that the code was executed on the OpenMP (CPU) host device, which
is exactly as expected since (1) we don't have any accelerator available on the login node
and (2) we only compiled a CPU target for the code. This means that if you run the same
binary on one of the GPU nodes, you will no longer see the `[hipSYCL Warning]` (since
CUDA drivers are now available), but you will still get the same program output `Chosen
device: hipSYCL OpenMP host device` (since the code is still only compiled for CPU targets).

### Compiling for Nvidia GPUs

The only thing we need to change when compiling for GPUs is to add new target options to
the compiler string. The only complicating issue here might be to figure out which target
architecture corresponds to the hardware at hand, but for the P100 GPUs on Saga the
name of the target should be `cuda:sm_60` (`cuda:sm_80` for Betzy's A100 cards):

```console
[me@login-1.SAGA ~]$ syclcc --hipsycl-targets='omp;cuda:sm_60' -O3 -o hello_world hello_world.cpp
clang-11: warning: Unknown CUDA version. cuda.h: CUDA_VERSION=11010. Assuming the latest supported version 10.1 [-Wunknown-cuda-version]
```

CUDA drivers is _not_ a prerequisite for compiling the CUDA target, so this can be done on
the login node. We see that we get a `Clang` warning due to the CUDA version, but this does
not seem to be a problem. The resulting executable can still be run on a pure CPU host (e.g.
the login node) with the same result as before:

```console
[me@login-1.SAGA ~]$ ./hello_world
[hipSYCL Warning] backend_loader: Could not load backend plugin: /cluster/software/hipSYCL/0.9.1-gcccuda-2020b/bin/../lib/hipSYCL/librt-backend-cuda.so
[hipSYCL Warning] libcuda.so.1: cannot open shared object file: No such file or directory
Chosen device: hipSYCL OpenMP host device
```

but if we instead run the code on a GPU node (here through an interactive job; remember to ask
for GPU resources on the `accel` partition) we see that the program is actually able to pick up
the GPU device:

```console
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --gpus-per-task=1 --partition=accel --mem=1G --pty bash
srun: job 3511513 queued and waiting for resources
srun: job 3511513 has been allocated resources
[me@c7-8.SAGA]$ ./hello_world
Chosen device: Tesla P100-PCIE-16GB 
```

Note that no code is actually executed on the device in this example, since the `sycl::queue`
remains empty, but at least we know that the hardware is visible to our application. Now the
next step will be to add some work that can be offloaded to the device.
