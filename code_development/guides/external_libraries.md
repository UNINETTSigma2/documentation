---
orphan: true
---

# Calling GPU accelerated libraries

One of the best ways to get the benefit of GPU acceleration is to call an
external library that is already accelerated. All of the major GPU hardware
vendors create such libraries and the advantage of using such a library is that
you will get the best performance possible for the available hardware. Examples
of GPU accelerated libraries include BLAS libraries such as [`cuBLAS` from
Nvidia](https://developer.nvidia.com/cublas), [`rocBLAS` from
AMD](https://rocblas.readthedocs.io/en/latest/) and [`oneMKL` from
Intel](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-math-kernel-library-onemkl.html).

One challenge with calling an external library is how to integrate with custom
accelerated code and how does one compile the code so that everything is
properly linked. To answer that this tutorial will go through:
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
vendors offers specialised BLAS routines for their own hardware. These
libraries offers the best in class performance and thanks to the shared
interface, abstracting over these different libraries are quite easy. We will
here show how to integrate with [`cuBLAS` from
Nvidia](https://developer.nvidia.com/cublas) which is compatible with the
hardware found on Saga and Betzy.

As an example we will use `cuBLAS` to perform a simple vector addition and then
calculate the sum of the vector in our own custom loop. The example is
simplistic to show how to combine `cuBLAS` and OpenACC, and our recommendation
is to always use BLAS libraries when performing mathematical computations.

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
device. Within this region we would normally perform calculations on the data,
but when integrating with `cuBLAS` we only need the address of the memory
(`#pragma acc host_data`). After the SAXPY routine is called we use the data to
calculate the sum as a normal OpenACC kernel.

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
load `CUDA` which contains the `cuBLAS` library which we will need to link in.

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

Finally we can run the program with the following call to `srun` (note that
this call works on both Saga and Betzy):

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
