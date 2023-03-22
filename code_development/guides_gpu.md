(dev-guides_gpu)=

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
