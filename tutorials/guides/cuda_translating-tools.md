---
orphan: true
---

(cuda2hip0sycl)=
# Translating GPU-accelerated applications

We present different tools to translate CUDA and OpenACC applications to target various GPU (Graphics Processing Unit) architectures (e.g. AMD and Intel GPUs). A special focus will be given to [`hipify`](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html), [`syclomatic`](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html#gs.o5pj6f) and [`clacc`](https://csmd.ornl.gov/project/clacc). These tools have been tested on the supercomputer [LUMI-G](https://lumi-supercomputer.eu/lumi_supercomputer/) in which the GPU partitions are of [AMD MI250X GPU](https://www.amd.com/en/products/server-accelerators/instinct-mi250x) type.

The aim of this tutorial is to guide users through a straightforward procedure for converting CUDA codes to HIP and SYCL, and OpenACC codes to OpenMP offloading. By the end of this tutorial, we expect users to learn about:

- How to use the `hipify-perl` and `hipify-clang` tools to translate CUDA sources to HIP sources.
- How to use the `syclomatic` tool to convert CUDA source to SYCL.
- How to use the `clacc` tool to convert OpenACC application to OpenMP offloading.
- How to compile the generated HIP, SYCL and OpenMP applications.

```{contents}
:depth: 2
```
(cuda2hip)=
## Translating CUDA to HIP with Hipify 

In this section, we cover the use of `hipify-perl` and `hipify-clang` tools to translate a CUDA application to HIP.

### Hipify-perl

The `hipify-perl` tool is a script based on perl that translates CUDA syntax into HIP syntax (see .e.g. [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl). As an example, in a CUDA code that makes use of the CUDA functions `cudaMalloc` and `cudaDeviceSynchronize`, the tool will replace `cudaMalloc` by the HIP function `hipMalloc`. Similarly for the CUDA function `cudaDeviceSynchronize`, which will be replaced by `hipDeviceSynchronize`. We list below the basic steps to run `hipify-perl`

- **Step 1**: loading modules

On LUMI-G, the following modules need to be loaded:

```console
$module load CrayEnv
```

```console
$module load rocm
```
- **Step 2**: generating `hipify-perl` script

```console
$hipify-clang --perl
```
- **Step 3**: running `hipify-perl`

```console
$perl hipify-perl program.cu > program.cu.hip
```
- **Step 4**: compiling with `hipcc` the generated HIP code

```console
$hipcc --offload-arch=gfx90a -o exec_hip program.cu.hip
```
Despite of the simplicity of the use of `hipify-perl`, the tool might not be suitable for large applications, as it relies heavily on substituting CUDA strings with HIP strings (e.g. it replaces *cuda* with *hip*). In addition, `hipify-perl` lacks the ability of [distinguishing device/host function calls](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl). The alternative here is to use `hipify-clang` as we shall describe in the next section.

(hipify-clang)=
### Hipify-clang

As described [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl), the `hipify-clang` tool is based on clang for translating CUDA sources into HIP sources. The tool is more robust for translating CUDA codes compared to the `hipify-perl` tool. Furthermore, it facilitates the analysis of the code by providing assistance.

In short, `hipify-clang` requires `LLVM+CLANG` and `CUDA`. Details about building `hipify-clang` can be found [here](https://github.com/ROCm-Developer-Tools/HIPIFY). Note that `hipify-clang` is available on LUMI-G. The issue however might be related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity container. Here we present a step-by-step guide for running `hipify-clang`:

- **Step 1**: pulling a CUDA singularity container e.g.

```console
$singularity pull docker://nvcr.io/nvidia/cuda:11.4.0-devel-ubi8
```
- **Step 2**: loading a ROCM module before launching the container.

```console
$ml rocm
```

During our testing, we used the rocm version `rocm-5.0.2`. 

- **Step 3**: launching the container

```console
$singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif
```

where the current directory `$PWD` in the host is mounted to that of the container, and the directory `/opt` in the host is mounted to the that inside the container.

- **Step 4**: setting the environment variable `$PATH`
In order to run `hipify-clang` from inside the container, one can set the environment variable `$PATH` that defines tha path to look for the binary `hipify-clang`

```console
$export PATH=/opt/rocm-5.0.2/bin:$PATH
```

- **Step 5**: running `hipify-clang`

```console
$hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include
```

Here the cuda path and the path to the *includes* and *defines* files should be specified. The CUDA source code and the generated output code are `program.cu` and `hip_program.cu.hip`, respectively.

- **Step 6**: the syntax for compiling the generated hip code is similar to the one described in the previous section (see the hipify-perl section).

(cuda2sycl)=
## Translating CUDA to SYCL with Syclomatic

[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is another conversion tool. However, instead of converting CUDA code to HIP syntax, SYCLomatic converts the code to SYCL/DPC++. The use of SYCLomatic requires CUDA libraries, which can be directly installed in an environment or it can be extracted from a CUDA container. Similarly to previous section, we use singularity container. Here is a step-by-step guide for using `SYCLamatic`

**Step 1** Downloading `SYCLomatic` e.g. the last release from [here](https://github.com/oneapi-src/SYCLomatic/releases)

```console
wget https://github.com/oneapi-src/SYCLomatic/releases/download/20230208/linux_release.tgz
```

**Step 2** Decompressing the tarball into a desired location:

```console
$tar -xvzf linux_release.tgz -C [desired install location]
```

**Step 3** Adding the the executable ```c2s``` which is located in ```[install location]/bin``` in your path, either by setting the environment variable `$PATH`

```console
$export PATH=[install location]/bin:$PATH
```

Or by creating a symbolic link into a local ```bin``` folder:

```console
$ln -s [install location]/bin/dpct /usr/bin/c2s
```

**Step 4** <a name="SYCLomatic_s_4"></a> Launching `SYCLomatic`. This is done by running `c2s` from inside a CUDA container. This is similar to steps 1, 3 and 5 in the previous {ref}`section <hipify-clang>`.

```console
$c2s [file to be converted]
```

This will create a folder in the current directory called ```dpct_output```, in which the converted file is generated.

**Step 5** Compiling the generated SYCL code


**_step 5.1_** Look for errors in the converted file

In some cases, `SYCLOmatic` might not be able to convert part of the code. In such cases, `SYCLomatyic` will comment on the parts it is unsure about. For example, these comments might look something like this:
```
/*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
*/
```
Before compiling, these sections will need to be manually checked for errors.

**_step 5.2_**
Once you have a valid file, you may compile it with the SYCL compiler of your choosing. There are many choices for such compilers, which vary based on the devices you are compiling for. Please confer with the [INTEL SYCL documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/compiling-sycl-with-different-gpus.html) if you are unsure what compiler to use.

*PS: Syclomatic generates data parallel C++ code (DPC++) in stead of a pure SYCL code. This means that you either need to manually convert the DPC++ code to SYCL if you want to use a pure SYCL compiler, or you need to use the intel OneAPI kit to compile the DPC++ code directly*

**_Compiling pure SYCL code_**
To compile the SYCL code on out clusters you need access to a SYCL compiler. On SAGA and BETZY this is straigthforward and is discussed in this tutorial: [What is SYCL](https://documentation.sigma2.no/code_development/guides/hipsycl.html). At the time of writing, LUMI does not have a global installation of ```hipSYCL```. We must therefore utilize easybuild to get access to it. To accsess to ```hipSYCL``` on LUMI use these terminal commands.

```
$export EBU_USER_PREFIX=/project/project_465000096/EasyBuild
$module load LUMI/22.08
$module load partition/G
$module load rocm
$module load hipSYCL/0.9.3-cpeCray-22.08
```
After this you should be able to follow the tutorial [mentioned above]((https://documentation.sigma2.no/code_development/guides/hipsycl.html))

### Launching SYCLomatic through a singularity container

An alternative to the steps mentioned above is to create a singularity .def file (see an example [here](./syclomatic_doc/syclomatic.def)). This can be done in the following:  

First, build a container image:

_OBS: In most systems, you need sudo privileges to build the container. You do not have this on our clusters, you should therefore consider building a container locally and then copying it over to the cluster using [scp](https://documentation.sigma2.no/getting_started/getting_started.html#transferring-files) or something similar._

```console
$singularity build syclomatic.sif syclomatic.def
```

Then execute the `SYCLomatic` tool from inside the container:

```console 
$singularity exec syclomatic.sif c2s [file to be converted]
```

This will create the same  ```dpct_output``` folder as mentioned in _step 4_.

## Translate OpenACC to OpenMP with Clacc

`Clacc` is a tool to translate `OpenACC` to `OpenMP` offloading with the Clang/LLVM compiler environment. In the following we present a step-by-step guide for building and using `Clacc`:

**_Step 1.1_**
Load the following modules to be able to build `Clacc` (For LUMI-G):

```console 
module load CrayEnv
module load rocm
```
**_Step 1.2_**
Build and install `Clacc.`
The building process will spend about 5 hours.

```console 
$ git clone -b clacc/main https://github.com/llvm-doe-org/llvm-project.git
$ cd llvm-project
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=../install     \
        -DCMAKE_BUILD_TYPE=Release            \
        -DLLVM_ENABLE_PROJECTS="clang;lld"    \
        -DLLVM_ENABLE_RUNTIMES=openmp         \
        -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" \
        -DCMAKE_C_COMPILER=gcc                \
        -DCMAKE_CXX_COMPILER=g++              \
        ../llvm
$ make
$ make install
```
**_Step 1.3_**
Set up environment variables to be able to work from /install directory, which is the easiest solution. For more advanced usage (ie. wanting to modify `Clacc`, see "Usage from Build directory": (https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md))

```console
$ export PATH=`pwd`/../install/bin:$PATH
$ export LD_LIBRARY_PATH=`pwd`/../install/lib:$LD_LIBRARY_PATH
```
**_Step 2_**
To compile the produced `OpenMP` code, you will need to load these modules:

```console
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
```
**_Step 2.1_**
Compile & run for host:
```console
$ clang -fopenacc openACC_code.c && ./output.out
```
**_Step 2.2_**
Compile & run on AMD GPU:
```console
$ clang -fopenacc -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a openACC_code.c && ./output.out
```
**_Step 2.3_**
Source to source mode with `OpenMP` port printed to console:
```console
$ clang -fopenacc-print=omp OpenACC.c
```
**_Step 3_**
Compile code with cc
```console
cc -fopenmp -o executable_name OpenMP.c
```

# Conclusion

We have presented an overview of the usage of available tools to convert CUDA codes to HIP and SYCL, and OpenACC codes to OpenMP offloading. In general the translation process for large applications might cover about 80% of the source code and thus requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process requires that applications are written correctly according to the CUDA and OpenACC syntaxes. 

# Relevant links

[Hipify GitHub](https://github.com/ROCm-Developer-Tools/HIPIFY)

[HIPify Reference Guide v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

[HIP example](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

[Porting CUDA to HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

[SYCLomatic Github](https://github.com/oneapi-src/SYCLomatic)

[Installing SYCLamatic](https://github.com/oneapi-src/SYCLomatic/releases)

[Clacc Main repository README](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)

