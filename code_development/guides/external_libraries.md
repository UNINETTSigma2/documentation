---
orphan: true
---

# Calling GPU accelerated libraries

One of the best ways to get the benefit of GPU acceleration is to call an
external library that is already accelerated. All of the major GPU hardware
vendors create such libraries and the advantage of their use is that you will
get the best performance possible for the available hardware. Examples of GPU
accelerated libraries include BLAS (Basic Linear Algebra Subprograms) libraries
such as [`cuBLAS` from Nvidia](https://developer.nvidia.com/cublas), [`rocBLAS`
from AMD](https://rocblas.readthedocs.io/en/latest/) and [`oneMKL` from
Intel](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-math-kernel-library-onemkl.html).

One challenge with calling an external library is related to its integration
with user accelerated code and how to compile the code so that everything is
linked. To address these issues this tutorial will go through:
- How to call different GPU accelerated libraries from both C/C++ and Fortran.
- How to combine external accelerated libraries and custom offloading code.
  - Focusing on OpenACC and OpenMP offloading
- How to compile your code so that the external libraries are linked.

```{contents}
:depth: 2
```

(cublas_openacc)=
## Calling `cuBLAS` from OpenACC

> The BLAS (Basic Linear Algebra Subprograms) are routines that provide
> standard building blocks for performing basic vector and matrix operations. -
> [netlib](https://www.netlib.org/blas/)

As noted in the introduction to this tutorial, all of the major GPU hardware
vendors provide specialised BLAS routines for their own hardware. These
libraries offers the best in class performance and thanks to the shared
interface, one can easily abstract over multiple libraries from different
vendors. Here we will show how to integrate OpenACC with [`cuBLAS` from
Nvidia](https://developer.nvidia.com/cublas). The `cuBLAS` library is a BLAS
implementation for Nvidia GPUs which is compatible with the hardware found on
Saga and Betzy.

As an example we will use `cuBLAS` to perform a simple vector addition and then
calculate the sum of the vector in our own custom loop. The example allows us
to show how to combine `cuBLAS` and OpenACC, and our recommendation is to
always use BLAS libraries when performing mathematical computations.

```{eval-rst}
.. literalinclude:: external_libraries/cublas/openacc.c
   :language: c
```
```{eval-rst}
:download:`cublas_openacc.c <./external_libraries/cublas/openacc.c>`
```

The main focus of our changes are in the following lines, where we call the
SAXPY routine within the already established OpenACC data region.

```{eval-rst}
.. literalinclude:: external_libraries/cublas/openacc.c
   :language: c
   :lines: 46-67
```

In the above section one can see that we first create an OpenACC data region
(`#pragma acc data`) so that our compute vectors are available on the GPU
device. Within this region we would normally have accelerated loops that do
calculations on the data, but when integrating with `cuBLAS` we only need the
address of the memory (`#pragma acc host_data`). After the SAXPY routine is
called we use the data to calculate the sum as a normal OpenACC kernel.

Combining `cuBLAS` and OpenACC in this manner allows us to call accelerated
libraries without having to perform low-level memory handling as one would
normally do with such a library.

---

To compile this code we will first need to load a few modules.

`````{tabs}
````{group-tab} Saga

```console
$ module load NVHPC/21.11 CUDA/11.4.1
```
````
````{group-tab} Betzy

```console
$ module load NVHPC/21.7 CUDA/11.4.1
```
````
`````

We first load `NVHPC` which contains the OpenACC C compiler (`nvc`), then we
load `CUDA` which contains the `cuBLAS` library which we will need to link to.

To compile we can use the following:

`````{tabs}
````{group-tab} Saga

```console
$ nvc -acc -Minfo=acc -gpu=cc60 -lcublas -o cublas_acc cublas_openacc.c
```
````
````{group-tab} Betzy

```console
$ nvc -acc -Minfo=acc -gpu=cc80 -lcublas -o cublas_acc cublas_openacc.c
```
````
`````

Finally, we can run the program using the `srun` command which works on both
Saga and Betzy:

```console
$ srun --account=nn<XXXX>k --ntasks=1 --time=02:00 --mem=1G --partition=accel --gpus=1 ./cublas_acc
srun: job <NNNNNN> queued and waiting for resources
srun: job <NNNNNN> has been allocated resources
Starting SAXPY + OpenACC program
  Initializing vectors on CPU
  Creating cuBLAS handle
  Starting calculation
  Calculation produced the correct result of '4 * 10000 == 40000'!
Ending SAXPY + OpenACC program
```

(cublas_openmp)=
## Calling `cuBLAS` from OpenMP offloading

OpenMP support offloading to GPUs in the same way as OpenACC. We will therefore
use the same example as above, but this time use OpenMP's offloading
capabilities.

Since the program has not changed much from above we have highlighted the major
differences from the OpenACC version.

```{eval-rst}
.. literalinclude:: external_libraries/cublas/omp.c
   :language: c
   :emphasize-lines: 7,46,54,63
```
```{eval-rst}
:download:`cublas_omp.c <./external_libraries/cublas/omp.c>`
```

As can be seen in the code above, our interaction with the `cuBLAS` library did
not have to change, we only had to change the directives we used to make the
compute vectors available. As with OpenACC, in OpenMP we start by creating a
data region to make our compute vectors accessible to the GPU (done with
`#pragma omp target data map(...)`). We then make the pointers to this data
available for our CPU code so that we can pass valid pointers to `cuBLAS`
(pointers made available with `#pragma omp target data use_device_ptr(...)`).
Finally we show that we can also use the vectors we uploaded in custom
offloading loops.

---

To compile the above OpenMP code we first need to load the necessary modules:

`````{tabs}
````{group-tab} Saga

```console
$ module load Clang/13.0.1-GCCcore-11.2.0-CUDA-11.4.1
```

Since the GPUs on Saga are a couple of generation older we can't use `NVHPC`
for OpenMP offloading. We instead use `Clang` to show that it works on Saga as
well.
````
````{group-tab} Betzy

```console
$ module load NVHPC/21.7 CUDA/11.4.1
```
````
`````

And then we compile with:

`````{tabs}
````{group-tab} Saga

```console
$ clang -o cublas_omp cublas_omp.c -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_60 -lcublas
```

Since the GPUs on Saga are a couple of generation older we can't use `NVHPC`
for OpenMP offloading. We instead use `Clang` to show that it works on Saga as
well.
````
````{group-tab} Betzy

```console
$ nvc -mp=gpu -Minfo=mp -gpu=cc80 -lcublas -o cublas_omp cublas_omp.c
```
````
`````

Finally we can run the program with the following call to `srun` (note that
this call works on both Saga and Betzy):

```console
$ srun --account=nn<XXXX>k --ntasks=1 --time=02:00 --mem=1G --partition=accel --gpus=1 ./cublas_omp
srun: job <NNNNNN> queued and waiting for resources
srun: job <NNNNNN> has been allocated resources
Starting SAXPY + OpenMP offload program
  Initializing vectors on CPU
  Creating cuBLAS handle
  Starting calculation
  Calculation produced the correct result of '4 * 10000 == 40000'!
Ending SAXPY + OpenMP program
```

(calling-cufft-from-openacc)=

## Calling cuFFT from OpenACC 

(summary)=

## Summary

In this section we provide an overview on how to implement a GPU-accelerated library FFT (Fast Fourier Transform) in an OpenACC application and the serial version of the FFTW library. Here we distinguish between two GPU-based FFT libraries: [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) and [cuFFTW](https://docs.nvidia.com/cuda/cufft/index.html#fftw-supported-interface). The cuFFT library is the NVIDIA-GPU based design, while cuFFTW is a porting version of the existing [FFTW](https://www.fftw.org/) library. In this tutorial, both libraries will be addressed with a special focus on the implementation of the cuFFT library. Specifically, the aim of this tutorial is to:

* Show how to incorporate the FFTW library in a serial code.
* Describe how to use the cuFFTW library.
* Show how to incorporate the cuFFT library in an OpenACC application interface.
* Describe briefly how to enable cuFFT to run on OpenACC stream.
* Describe the compilation process of FFTW and cuFFT.

The implementation will be illustrated for a one-dimensional (1D) scenario and will be further described for 2D and 3D cases.

```{contents} Table of Contents
```

(generality-of-fft)=

## Generality of FFT

In general, the implementation of an FFT library is based on three major steps as defined below:

- Creating plans (initialization).

- Executing plans (create a configuration of a FFT plan having a specified dimension and data type).

- Destroying plans (to free the ressources associated with the FFT plans).

These steps necessitate specifying the direction, in which the FFT algorithm should be performed: forward or backward (or also inverse of FFT), and the dimension of the problem at hands as well as the precision (i.e. double or single precision); this is in addition to the nature of the data (real or complex) to be transformed.  

In the following, we consider a one-dimensional (1D) scenario, in which the execution is specified for a double precision complex-to-complex transform plan in the forward and backward directions. The implementation is illustrated via a Fortran code. The latter can be adjusted to run calculations of a single precision as well as of real-to-real/complex transform and can be further extended to multi-dimension cases (i.e. 2D and 3D). We first start with the FFT implementation in a serial-CPU scheme and further extend it to a GPU-accelerated case.  The implementation is illustrated for a simple example of a function defined in time-domain. Here we choose a sinus function (i.e. f(t)=sin(&omega;t) with &omega; is fixed at the value 2), and its FFT should result in a peak around the value &omega;=2 in the frequency domain.  

(implementation-of-fftw)=

## Implementation of FFTW   

The implementation of the FFTW library is shown below and a detailed description of the library can be found [here](https://www.fftw.org/).

As described in the code, one needs to initialize the FFT by creating plans. Executing the plans requires specifying the transform direction: *FFTWFORWARD* for the forward direction or *FFTWBACKWARD* for the backward direction (inverse FFT). These two parameters should be defined as an integer parameter. An alternative is to include the `fftw3.f` file as a header (i.e. `include "fftw3.f"`), which contains all parameters required for a general use of FFTW. In the case the file is included, the value of the direction parameter does not need to be defined.  

The argument *FFTW_MEASURE* in the function `dfftw_plan_dft_1d` means that FFTW measures the execution time of several FFTs in order to find the optimal way to compute the FFT, which might be time-consuming. An alternative is to use *FFTW_ESTIMATE*, which builds a reasonable plan without any computation. This procedure might be less optimal (see [here](https://www.fftw.org/) for further details). 

Note that when implementing the FFTW library, the data obtained from the backward direction need to be normalized by dividing the output array by the size of the data, while those of forward direction do not. This is only valid when using the FFTW library.

To check the outcome of the result in the forward direction, one can plot the function in the frequency-domain, which should display a peak around the value &omega;=+2 and -2 as the function is initially symmetric. By performing the backward FFT of the obtained function, one should obtain the initial function displayed in time-domain (i.e. sin(2t)). This checking procedure holds also when implementing a GPU version of the FFT library.

For completeness, porting the FFTW library to [cuFFTW](https://docs.nvidia.com/cuda/cufft/index.html#fftw-supported-interface) does not require modifications in the code - it is done by replacing the file `fftw3.h` with `cufftw.h`.

``` fortran
      module parameter_kind
        implicit none
        public
        integer, parameter :: FFTW_FORWARD=-1,FFTW_BACKWARD=+1
        integer, parameter :: FFTW_MEASURE=0
        integer, parameter :: sp = selected_real_kind(6, 37) !Single precision
        integer, parameter :: dp = selected_real_kind(15, 307) !Double precision
        integer, parameter :: fp = dp
        real(fp), parameter :: pi = 4.0_fp*atan(1.0_fp),dt=0.25_fp
      end module parameter_kind

      program fftw_serial

       use parameter_kind

       implicit none

       !include "fftw3.f"

       integer, parameter   :: nt=512
       integer              :: i,ierr
       integer*8            :: plan_forward,plan_backward
       complex(fp), allocatable  :: in(:),out(:),f(:)
       real(fp), allocatable     :: t(:),w(:)

       allocate(t(nt),w(nt)); allocate(f(nt))

       call grid_1d(nt,t,w)

!Example of sine function
       do i=1,nt
          f(i) = cmplx(sin(2.0_fp*t(i)),0.0_fp)
       enddo

       print*,"--sum before FFT", sum(real(f(1:nt/2)))

!Creating 1D plans
       allocate(in(nt),out(nt))
       call dfftw_plan_dft_1d(plan_forward,nt,in,out,FFTW_FORWARD,FFTW_MEASURE)
       call dfftw_plan_dft_1d(plan_backward,nt,in,out,FFTW_BACKWARD,FFTW_MEASURE)

!Forward FFT
       in(:) = f(:)
       call dfftw_execute_dft(plan_forward, in, out)
       f(:) = out(:)

!Backward FFT
       call dfftw_execute_dft(plan_backward, out, in)
!The data on the backforward are unnormalized, so they should be divided by N.        
       in(:) = in(:)/real(nt)

!Destroying plans
       call dfftw_destroy_plan(plan_forward)
       call dfftw_destroy_plan(plan_backward)

       print*,"--sum iFFT", sum(real(in(1:nt/2)))

!Printing the FFT of sin(2t)
        do i=1,nt/2
           write(204,*)w(i),dsqrt(cdabs(f(i))**2)
        enddo
        deallocate(in); deallocate(out); deallocate(f)
      end

       subroutine grid_1d(nt,t,w)
        use parameter_kind

        implicit none
        integer             :: i,nt
        real(fp)            :: t(nt),w(nt)

!Defining a uniform temporal grid
       do i=1,nt
          t(i) = (-dble(nt-1)/2.0_fp + (i-1))*dt
       enddo

!Defining a uniform frequency grid
       do i=0,nt/2-1
          w(i+1) = 2.0_fp*pi*dble(i)/(nt*dt)
       enddo
       do i=nt/2,nt-1
          w(i+1) = 2.0_fp*pi*dble(i-nt)/(nt*dt)
       enddo
       end subroutine grid_1d
```

(compilation-process-of-fftw)=

## Compilation process of FFTW

The FFTW library should be linked with fftw3 (i.e. `-lfftw3`) for the double precision, and fftw3f (i.e. `-lfftw3f`) for the single precision case. 

Here is an example of a module to be loaded. 

On Saga:
```console
$ module load FFTW/3.3.9-intel-2021a
```
The same module is available on Betzy.

To compile: 
```console
$ ifort -lfftw3 -o fftw.serial fftw_serial.f90
```
   
In the case of using the cuFFTW library, the linking in the compilation syntaxt should be provided for both cuFFT and cuFFTW libraries.

(implementation-of-cufft)=

## Implementation of cuFFT 

We consider the same scenario as described in the previous section but this time the implementation involves the communication between a CPU-host and GPU-device by calling the cuFFT library. The cuFFT implementation is shown below. 

Similarly to the FFTW library, the implementation of the GPU-accelerated cuFFT library is conceptually based on creating plans, executing and destroying them. The difficulty here however is how to call the cuFFT library, which is written using a low-level programming model, from an OpenACC application interface. In this scenario, there are steps that are executed by the cuFFT library and other steps are executed by OpenACC kernels. Executing all these steps requires sharing data. In other words, it requires making OpenACC aware of the GPU-accelerated cuFFT library. This is done in OpenACC by specifying the directive `host_data` together with the clause `use_device(list-of-arrays)`. This combination permits to access the device address of the listed arrays in the `use_device()` clause from the [host](https://www.nvidia.com/docs/IO/116711/OpenACC-API.pdf). The arrays, which should be already present on the device memory, are in turn passed to the cuFFT functions (i.e. `cufftExecZ2Z()` in our example). The output data of these functions is not normalized, and thus it requires to be normalized by dividing by the size of the array. The normalisation may be followed by the function `cufftDestroy()` to free all GPU resources associated with a cuFFT plan and destroy the internal plan data structure.

It is worth noting that the cuFFT library uses CUDA streams for an asynchronous execution, which is not the case for OpenACC. It is therefore necessary to make the cuFFT runs on OpenACC streams. This is done by calling the routine `cufftSetStream()`, which is part of the cuFFT module. The routine includes the function `acc_get_cuda_stream()`, which enables identifying the CUDA stream.

Note that the use of the OpenACC runtime routines and the cuFFT routines requires including the header lines `use openacc` and `use cufft`.

The tables below summarize the calling functions in  the case of a multi-dimension data having a simple or double complex data type (see [here](https://docs.nvidia.com/hpc-sdk/compilers/fortran-cuda-interfaces/index.html) for more details).

``` fortran
     module parameter_kind
        implicit none
        public
        integer, parameter :: sp = selected_real_kind(6, 37)   !Single precision
        integer, parameter :: dp = selected_real_kind(15, 307) !Double precision
        integer, parameter :: fp = dp
        real(fp), parameter :: pi = 4.0_fp*atan(1.0_fp),dt=0.25_fp
      end module parameter_kind

      program cufft_acc

       use parameter_kind
       use cufft
       use openacc

       implicit none

       integer, parameter   :: nt=512
       integer              :: i,ierr,plan
       complex(fp), allocatable  :: in(:),out(:)
       real(fp), allocatable     :: t(:),w(:)

       allocate(t(nt),w(nt)); allocate(in(nt),out(nt))

       call grid_1d(nt,t,w)

!Example of a sinus function
       do i=1,nt
          in(i) = cmplx(sin(2.0_fp*t(i)),0.0_fp)
       enddo

       print*,"--sum before FFT", sum(real(in(1:nt/2)))
!cufftExecZ2Z executes a double precision complex-to-complex transform plan
       ierr = cufftPlan1D(plan,nt,CUFFT_Z2Z,1)
       
!acc_get_cuda_stream: tells the openACC runtime to identify the CUDA
!stream used by CUDA
       ierr = ierr + cufftSetStream(plan,acc_get_cuda_stream(acc_async_sync))

!$acc data copy(in) copyout(out)
!$acc host_data use_device(in,out)
        ierr = ierr + cufftExecZ2Z(plan, in, out, CUFFT_FORWARD)
        ierr = ierr + cufftExecZ2Z(plan, out, in, CUFFT_INVERSE)
!$acc end host_data 

!$acc kernels
       out(:) = out(:)/nt
       in(:) = in(:)/nt
!$acc end kernels
!$acc end data

       ierr =  ierr + cufftDestroy(plan)
       
       print*,""
       if(ierr.eq.0) then
         print*,"--Yep it works :)"
       else
         print*,"Nop it fails, I stop :("
       endif
       print*,""
       print*,"--sum iFFT", sum(real(in(1:nt/2)))

!printing the fft of sinus
        do i=1,nt/2
           write(204,*)w(i),sqrt(cabs(out(i))**2)
        enddo
        deallocate(in); deallocate(out)
      end

      subroutine grid_1d(nt,t,w)
        use parameter_kind

        implicit none
        integer             :: i,nt
        real(fp)            :: t(nt),w(nt)

!Defining a uniform temporal grid
       do i=1,nt
          t(i) = (-dble(nt-1)/2.0_fp + (i-1))*dt
       enddo

!Defining a uniform frequency grid
       do i=0,nt/2-1
          w(i+1) = 2.0_fp*pi*dble(i)/(nt*dt)
       enddo
       do i=nt/2,nt-1
          w(i+1) = 2.0_fp*pi*dble(i-nt)/(nt*dt)
       enddo
     end subroutine grid_1d
```

Dimension| Creating a FFT plan|
--- |--- |
1D | cufftPlan1D(plan,nx, FFTtype,1) | 
2D | cufftPlan2d( plan, ny, nx, FFTtype) |
3D | cufftPlan3d( plan, nz, ny, nx, FFTtype ) |

**Table 1.** *Creating FFT plans in 1D, 2D and 3D. Dimension executing a double precision complex-to-complex transform plan in the forward and backward directions. Nx is the size of a 1D array, nx and ny the size of a 2D array, and nx,ny,nz is the size of a 3D array. The FFTtype specifies the data type stored in the arrays in and out as described in the **Table 2**.*


Precision of the transformed plan | subroutine | FFTtype |
--- | --- | --- |
Double precision complex-to-complex | cufftExecZ2Z( plan, in, out, direction ) | ”CUFFT_Z2Z” |
Single precision complex-to-complex | cufftExecC2C( plan, in, out, direction ) | ”CUFFT_C2C” |


**Table 2.** *The execution of a function using the cuFFT library. The direction specifies the FFT direction: “CUFFT_FORWARD” for forward FFT and “CUFFT_INVERSE” for backward FFT. The input data are stored in the array in, and the results of FFT for a specific direction are stored in the array out.*
 
(compilation-process-of-cufft)=

## Compilation process of cuFFT

The cuFFT library is part of the CUDA toolkit, and thus it is supported by the NVIDIA-GPU compiler. Therefore, the only modules are required to be load are NVHPC and CUDA modules.
 
Modules to be loaded:

`````{tabs}
````{group-tab} Saga

```console
$ module load NVHPC/21.11 CUDA/11.4.1
```
````
````{group-tab} Betzy

```console
$ module load NVHPC/21.7 CUDA/11.4.1
```
````
`````

We compile using the NVIDIA Fortran compiler nvfortran. The compilation process requires linking the cuFFT library (`-lcufft`) and adding the CUDA version library to the syntax of the compilation (`-cudalib=cufft`). 

```console
$ nvfortran -lcufft -cudalib=cufft -acc -Minfo=accel -o cufft.acc cufft_acc.f90
```
Here the flag `-acc` enables OpenACC on NVIDIA-GPU. It is possible to specify the compute capability e.g. `-gpu=cc80` for Betzy and `-gpu=cc60` for Saga.

To run:
```console
$ srun --partition=accel --gpus=1 --time=00:01:00 --account=nnXXXXX --qos=devel --mem-per-cpu=1G ./cufft.acc
```

(conclusion)=

## Conclusion

In conclusion, we have provided a description on the implementation of the FFTW library in a serial code and the GPU-accelerated cuFFT targeting NVIDIA in an OpenACC application. The latter implementation illustrates the capability of calling a GPU-accelerated library written in a low-level programming model from an OpenACC application interface. Although the implementation has been done for a 1D problem, an extension to 2D and 3D scenarios is straightforward.   

