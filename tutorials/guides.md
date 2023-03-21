(dev-guides)=
# GPU programming models
- Introduction to GPU:
    - [<span style="color:green">Beginner</span>]
        [Introduction to using GPU partition](guides/gpu.md)
    - [<span style="color:green">Beginner</span>]
        [Offloading to GPU](guides/offloading.md)
- Calling GPU accelerated libraries:
    - [<span style="color:green">Beginner</span>]
        {ref}`Calling cuBLAS from OpenACC<cublas_openacc>`
    - [<span style="color:green">Beginner</span>]
        {ref}`Calling cuBLAS from OpenMP<cublas_openmp>`
    - [<span style="color:green">Beginner</span>]
        {ref}`Calling cuFFT from OpenACC<cufft_openacc>`    
- GPU programming with OpenACC:
    - [<span style="color:green">Beginner</span>]
        [Getting started with OpenACC and Nvidia Nsight](guides/openacc.md)
    - [<span style="color:orange">Intermediate</span>]
        [Async and Multi-GPU OpenACC](guides/async_openacc.md)
- GPU programming with OpenMP:
    - [<span style="color:green">Beginner</span>]
        [Introduction to OpenMP offloading](guides/ompoffload.md)
- GPU programming with SYCL:
    - [<span style="color:green">Beginner</span>]
        [Getting started with hipSYCL](guides/hipsycl.md)
    - [<span style="color:green">Beginner</span>]
        [SYCL Academy tutorial](guides/sycl_academy.md)
    - [<span style="color:green">Beginner</span>]
        [SYCL ENCCS tutorial](guides/sycl_enccs.md)
    - [<span style="color:orange">Intermediate</span>]
        [Unified Shared Memory with SYCL](guides/sycl_usm.md)
- Porting applications:
    - [<span style="color:green">Beginner</span>]
        [Porting OpenACC to OpenMP offloading](guides/converting_acc2omp/openacc2openmp.md) 
     - [<span style="color:green">Beginner</span>]
        {ref}`Translating CUDA to HIP with Hipify<cuda2hip>`
    - [<span style="color:green">Beginner</span>]
        {ref}`Translating CUDA to SYCL with Syclomatic<cuda2sycl>`  
    - [<span style="color:green">Beginner</span>]
        {ref}`Translating OpenACC to OpenMP with Clacc<acc2omp>`    
- Hybrid programming
    - [<span style="color:green">Beginner</span>]
        [MPI and OpenACC](guides/openacc_mpi.md)
    - [<span style="color:green">Intermediate</span>]
        [GPU-aware MPI with OpenACC and OpenMP](guides/gpuaware_mpi.md)            
- Offloading to GPU using Fortran 2008:
    - [<span style="color:green">Beginner</span>]
        [Offloading to GPU using Fortran 2008](guides/offloading-using-fortran.md)

# Machine Learning
- TensorFlow on GPU
    - [<span style="color:green">Beginner</span>]
        [Introduction to TensorFlow: part I](guides/tensorflow_gpu.md)
    - [<span style="color:green">Beginner</span>]
        [Introduction to TensorFlow: part II](guides/gpu/tensorflow.md)

# Containers with GPU support
- Building containers with Singularity:
    - [<span style="color:green">Beginner</span>]
        [Containers on NRIS HPC systems](guides/containers.md)
    - [<span style="color:green">Beginner</span>]
        [BigDFT with MPI and CUDA](guides/containers/bigdft.md)
    - [<span style="color:green">Beginner</span>]
        [Container with build environment](guides/container_env.md)
    - [<span style="color:green">Beginner</span>]
        [Container with MPI support](guides/container_mpi.md)
    - [<span style="color:green">Beginner</span>]
        [Container with GPU support (OpenACC)](guides/container_openacc.md)
    - [<span style="color:green">Beginner</span>]
        [CUDA Container](guides/gpu/cuda-container.md)

# Monitoring GPU accelerated applications
- Profiling and debugging CUDA applications
    - [<span style="color:green">Beginner</span>]
        [Stencil Communication Pattern with CUDA](guides/stencil.md)
