# Translating GPU-accelerated applications

We present different tools to translate CUDA-based codes to target various GPU (Graphics Processing Unit) architectures (e.g. AMD and Intel GPUs). A special focus will be to cover the following tools: (i) [`hipify`](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html); (ii) [`syclomatic`](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html#gs.o5pj6f), and (iii) [`clacc`](https://csmd.ornl.gov/project/clacc). These tools have been tested on the supercomputer [LUMI-G](https://lumi-supercomputer.eu/lumi_supercomputer/) in which the GPU partitions are of type [AMD MI250X GPU](https://www.amd.com/en/products/server-accelerators/instinct-mi250x).

The aim of this document is to guide users through a straightforward procedure for converting CUDA-based codes to other GPU-programming models, mainly, HIP, SYCL and OpenMP offloading. By the end of this document, we expect users to learn about:

- How to use the `hipify-perl` and `hipify-clang` tools to translate CUDA sources to HIP sources.
- How to use the `syclomatic` and `DPC++` tools to convert CUDA source to SYCL.
- How to use the `clacc` tool to convert OpenACC application to OpenMP offloading.
- How to compile the generated HIP, SYCL and OpenMP applications.

## Hipify 

In this section, we describe how to use `hipify-perl` and `hipify-clang` tools to translate a CUDA application to HIP.

### Hipify-perl

The `hipify-perl` tool is a script based on perl that translates cuda syntax into hip syntaxt (see .e.g. [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl). We list below the basic steps to run `hipify-perl`

- **Step 1**: load modules

On LUMI-G, the following modules need to be loaded:

`$module load CrayEnv`

`$module load rocm`

- **Step 2**: generate `hipify-perl` script

`$hipify-clang --perl`

- **Step 3**: run `hipify-perl`

`$perl hipify-perl program.cu > program.cu.hip`

- **Step 4**: compile with `hipcc`

`$hipcc --offload-arch=gfx90a -o exec_hip program.cu.hip` 

Despite of the simplicity of the use of `hipify-perl`, the tool might not be suitable for large applicatons, as it relies heavily on translating regular expressions (e.g. it replaces *cu* with *hip*). The alternative here is to use `hipify-clang` as described in the next section.

### Hipify-clang

As described [here](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl), the `hipify-clang` tool is based on clang for translating CUDA sources into HIP sources. In short, `hipify-clang` requires `LLVM+CLANG` and `CUDA`. Details about building `hipify-clang` can be found [here](https://github.com/ROCm-Developer-Tools/HIPIFY). Note that `hipify-clang` is available on LUMI-G. The issue however might be related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity container. In the following, we describe the necessary steps to run `hipify-clang`:

- **Step 1**: pull a cuda singularity container e.g.

`$singularity pull docker://nvcr.io/nvidia/cuda:11.4`

- **Step 2**: load a rocm module before launching the container.

`$ml rocm`

During our testing, we used the rocm version `rocm-5.0.2`. 

- **Step 3**: launch the container

`$singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif`

where the current directory $PWD in the host is mounted to the one of the container, and the directory `/opt` in the host is mounted to the one inside the container.

- **Step 4**: set the environment variable `$PATH`
In order to run `hipify-clang` from inside the container, one can set the environment variable `$PATH` that defines tha path to look for the binary `hipify-clang`

`$export PATH=/opt/rocm-5.0.2/bin:$PATH`

- **Step 5**: run `hipify-clang`

`$hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include`

Here the cuda path and the path to the includes and defines files should be specified. The CUDA source code and the generated output code are `program.cu` and `hip_program.cu.hip`, respectively.

- **Step 6**: the syntax for compiling the generated hip code is similar to the one described in the previous section (see hipify-per).

Some refs.

- https://github.com/ROCm-Developer-Tools/HIPIFY

- https://olcf.ornl.gov/wp-content/uploads/hip_for_cuda_programmers_slides.pdf

- https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition

- https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP

https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html

## Syclomatic

## Clacc

# Conclusion

We have presented an overview of the usage of available tools to convert CUDA-based applications to HIP, SYCL and OpenMP offloading (for OpenACC C source). In general the translation process for large applications covers about 80-90% of the source code and thus requires manual modification to complet the porting application. It is however worth noting that the accuracy of the translation process requires that applications are written correctly according to the cuda syntax. 

