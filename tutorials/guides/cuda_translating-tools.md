---
orphan: true
---

(cuda2hip0sycl)=
# Translating GPU-accelerated applications

We present different tools to translate CUDA-based codes to target various GPU (Graphics Processing Unit) architectures (e.g. AMD and Intel GPUs). A special focus will be put on [`hipify`](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html) and [`syclomatic`](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html#gs.o5pj6f). These tools have been tested on the supercomputer [LUMI-G](https://lumi-supercomputer.eu/lumi_supercomputer/) in which the GPU partitions are of [AMD MI250X GPU](https://www.amd.com/en/products/server-accelerators/instinct-mi250x) type.

The aim of this tutorial is to guide users through a straightforward procedure for converting CUDA-based codes to other GPU-programming models, mainly, HIP and SYCL. By the end of this tutorial, we expect users to learn about:

- How to use the `hipify-perl` and `hipify-clang` tools to translate CUDA sources to HIP sources.
- How to use the `syclomatic` and `DPC++` tools to convert CUDA source to SYCL.
- How to compile the generated HIP and SYCL applications.

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
Despite of the simplicity of the use of `hipify-perl`, the tool might not be suitable for large applicatons, as it relies heavily on translating regular expressions (e.g. it replaces *cuda* with *hip*). The alternative here is to use `hipify-clang` as described in the next section.

### Hipify-clang

As described [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl), the `hipify-clang` tool is based on clang for translating CUDA sources into HIP sources. In short, `hipify-clang` requires `LLVM+CLANG` and `CUDA`. Details about building `hipify-clang` can be found [here](https://github.com/ROCm-Developer-Tools/HIPIFY). Note that `hipify-clang` is available on LUMI-G. The issue however might be related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity container. Here we present a step-by-step guide to runing `hipify-clang`:

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

where the current directory $PWD in the host is mounted to that of the container, and the directory `/opt` in the host is mounted to the that inside the container.

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

- **Step 6**: the syntax for compiling the generated hip code is similar to the one described in the previous section (see hipify-per).

(cuda2sycl)=
## Translating CUDA to SYCL with Syclomatic

[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is another conversion tool. However, instead of converting CUDA code to HIP syntax, SYCLomatic converts the code to SYCL/DPC++. The use of SYCLomatic requires CUDA libraries, which can be directly installed in an environment or can be extracted from a CUDA container. Similarly to previous section, we use singularity container. Here is a step-by-step guide for using `SYCLamatic`

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

**Step 4** <a name="SYCLomatic_s_4"></a> Launching `SYCLomatic`

```console
$c2s [file to be converted]
```

This will create a folder in the current directory called ```dpct_output```, in which the converted file is generated.

**Step 5** Compiling the generated SYCL code


**_step 5.1_** Look for errors in the converted file

In some cases, SYCLOmatic might not be able to convert part of the code. In such cases, SYCLomatyic will comment on the parts it is unsure about. For example, these comments might look something like this:
```
/*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
*/
```
Before compiling, these sections will need to be manually checked for errors.

**_step 5.2_**
Once you have a valid file, you may compile it with the SYCL compiler of your choosing. There are many choices for such compilers, which vary based on the devices you are compiling for. Please confer with the [INTEL SYCL documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/compiling-sycl-with-different-gpus.html) if you are unsure what compiler to use.

### Launching SYCLomatic through a singularity container

An alternative to the steps mentioned above is to create a singularity .def file (see an example [here](./syclomatic_doc/syclomatic.def)). This can be done in the following:  

First, build a container image:

```console
$singularity build syclomatic.sif syclomatic.def
```

Then execute the `SYCLomatic` tool from inside the container:

```console 
$singularity exec syclomatic.sif c2s [file to be converted]
```

This will create the same  ```dpct_output``` folder as mentioned in _step 4_.


# Conclusion

We have presented an overview of the usage of available tools to convert CUDA-based applications to HIP and SYCL. In general the translation process for large applications might cover about 80% of the source code and thus requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process requires that applications are written correctly according to the CUDA syntax. 

# Relevant links

[Hipify GitHub](https://github.com/ROCm-Developer-Tools/HIPIFY)

[HIPify Reference Guide v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

[HIP example](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

[Porting CUDA to HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

[SYCLomatic Github](https://github.com/oneapi-src/SYCLomatic)

[Installing SYCLamatic](https://github.com/oneapi-src/SYCLomatic/releases)

