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

## Hipify 

In this section, we cover the use of `hipify-perl` and `hipify-clang` tools to translate a CUDA application to HIP.

### Hipify-perl

The `hipify-perl` tool is a script based on perl that translates CUDA syntax into HIP syntax (see .e.g. [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl). As an example, in a CUDA code that makes use of the CUDA functions `cudaMalloc` and `cudaDeviceSynchronize`, the tool will replace `cudaMalloc` by the HIP function `hipMalloc`. Similarly for the CUDA function `cudaDeviceSynchronize`, which will be replaced by `hipDeviceSynchronize`. We list below the basic steps to run `hipify-perl`

- **Step 1**: loading modules

On LUMI-G, the following modules need to be loaded:

`$module load CrayEnv`

`$module load rocm`

- **Step 2**: generating `hipify-perl` script

`$hipify-clang --perl`

- **Step 3**: running `hipify-perl`

`$perl hipify-perl program.cu > program.cu.hip`

- **Step 4**: compiling with `hipcc` the generated HIP code

`$hipcc --offload-arch=gfx90a -o exec_hip program.cu.hip` 

Despite of the simplicity of the use of `hipify-perl`, the tool might not be suitable for large applicatons, as it relies heavily on translating regular expressions (e.g. it replaces *cuda* with *hip*). The alternative here is to use `hipify-clang` as described in the next section.

### Hipify-clang

As described [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl), the `hipify-clang` tool is based on clang for translating CUDA sources into HIP sources. In short, `hipify-clang` requires `LLVM+CLANG` and `CUDA`. Details about building `hipify-clang` can be found [here](https://github.com/ROCm-Developer-Tools/HIPIFY). Note that `hipify-clang` is available on LUMI-G. The issue however might be related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity container. Here we present a step-by-step guide to runing `hipify-clang`:

- **Step 1**: pulling a CUDA singularity container e.g.

`$singularity pull docker://nvcr.io/nvidia/cuda:11.4.0-devel-ubi8`

- **Step 2**: loading a ROCM module before launching the container.

`$ml rocm`

During our testing, we used the rocm version `rocm-5.0.2`. 

- **Step 3**: launching the container

`$singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif`

where the current directory $PWD in the host is mounted to that of the container, and the directory `/opt` in the host is mounted to the that inside the container.

- **Step 4**: setting the environment variable `$PATH`
In order to run `hipify-clang` from inside the container, one can set the environment variable `$PATH` that defines tha path to look for the binary `hipify-clang`

`$export PATH=/opt/rocm-5.0.2/bin:$PATH`

- **Step 5**: running `hipify-clang`

`$hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include`

Here the cuda path and the path to the *includes* and *defines* files should be specified. The CUDA source code and the generated output code are `program.cu` and `hip_program.cu.hip`, respectively.

- **Step 6**: the syntax for compiling the generated hip code is similar to the one described in the previous section (see hipify-per).

## Syclomatic



# Conclusion

We have presented an overview of the usage of available tools to convert CUDA-based applications to HIP and SYCL. In general the translation process for large applications might cover about 80% of the source code and thus requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process requires that applications are written correctly according to the CUDA syntax. 

# Relevant links

[Hipify GitHub](https://github.com/ROCm-Developer-Tools/HIPIFY)

[HIPify Reference Guide v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

[HIP example](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

[Porting CUDA to HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

