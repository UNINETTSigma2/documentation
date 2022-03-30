# Offloading to GPUs

In high-performance computing offloading is the act of moving a computation
kernel from the main processor to one or more accelerators. In many cases the
**the computational kernel does not need to be programmed** but can be a
standard `for` (or `do` in Fortran) loop.

This document shows how to use the standard compilers available on {ref}`saga`
and {ref}`betzy` to offload computation to the attached GPUs. This document is
not considered as a comprehensive for how to perform offloading, but rather as
rather a compendium on the different compiler flags required to offload with
different compilers. For guides on how to transition your code to become
offload capable please {ref}`see our guides <dev-guides>`.

Below we have listed the necessary flags to enable GPU offloading for the
different systems NRIS users have access to. Both {ref}`saga` and {ref}`betzy`
are Nvidia systems, while {ref}`lumi` is an AMD based system. For information
about the specific accelerators, and their specific IDs often needed when
compiling, consult the below tabs.

````{tabs}
```{tab} Betzy

Betzy has `Nvidia A100` accelerators which support CUDA version `8.0`. The
identifier for the GPU is either `sm_80` or `cc80` depending on the compiler.
```
```{tab} Saga

Saga has `Nvidia P100` accelerators which support CUDA version `6.0`. The
identifier for the GPU is either `sm_60` or `cc60` depending on the compiler.
```
```{tab} LUMI-G

LUMI-G has `AMD MI250X` accelerators which is supported by ROCm. The identifier
for the GPU is `gfx908`.
```
````

## OpenMP

OpenMP gained support for accelerator offloading in version `4.0`. Most
compilers that support version `4.5` and above should be able to run on
attached GPUs. However, their speed can vary widely so it is recommended to
compare the performance.

If you are interested in learning more about OpenMP offloading we have
{ref}`a beginner tutorial on the topic here<ompoffload>`.

```{warning}
NVHPC does not support OpenMP offloading on {ref}`saga` as the generation of
GPUs on {ref}`saga` is older than what NVHPC supports. Thus, NVHPC _only_
supports OpenMP offloading on {ref}`betzy`.
```

```````{tabs}

``````{tab} Clang

`````{tabs}

````{tab} Nvidia

```bash
-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_<XX>
```
````
````{tab} AMD

```bash
-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx<XXX>
```
````
`````
``````
``````{tab} GCC

`````{tabs}

````{tab} Nvidia

```bash
-fopenmp -foffload=nvptx-none="-misa=sm_35"
```
````
````{tab} AMD

```bash
-fopenmp -foffload=amdgcn-amdhsa="-march=gfx<XXX>
```
````
`````
``````
``````{tab} NVHPC

`````{tabs}

````{tab} Nvidia

```bash
-mp=gpu -Minfo=mp,accel -gpu=cc<XX>
```
````
`````
``````
```````

## OpenACC

OpenACC is another open standard for supporting offloading to accelerators.
Since OpenACC was initially developed by Nvidia the best support for OpenACC is
found using Nvidia's compilers. However, several other compilers also support
OpenACC to some extent.

If you are interested in learning more about OpenACC offloading we have
{ref}`a beginner tutorial on the topic here<openacc>`.

```````{tabs}

``````{tab} GCC

`````{tabs}

````{tab} Nvidia

```bash
-fopenacc -foffload=nvptx-none="-misa=sm_35"
```
````
````{tab} AMD

```bash
-fopenacc -foffload=amdgcn-amdhsa="-march=gfx<XXX>
```
````
`````
``````
``````{tab} NVHPC

`````{tabs}

````{tab} Nvidia

```bash
-acc -Minfo=accel -gpu=cc<XX>
```
````
`````
``````
```````

## Standard Parallelism

Nvidia additionally supports offloading based on "Standard Parallelism" which
is capable of accelerating C++ `std::algorithms` and Fortran's `do concurrent`
loops.

You can read more about accelerating Fortran using `do concurrent`
{ref}`in our guide<offload_fortran_concurrent>`.

`````{tabs}
````{tab} NVHPC

```bash
-stdpar=gpu -Minfo=stdpar
```
````
`````
