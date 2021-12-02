---
orphan: true
---

# Offloading to GPU using Fortran 2008

## Introduction

In 2010 the ISO Standard Fortran 2008 introduced the `do concurrent` construct
which allows to express loop-level parallelism. The current compilers support multicore 
parallel execution, but presently only NVIDIA offer a compiler that support offload to their GPUs.

The NVIDIA HPC Fortran (Formerly PGI Fortran) supports using the `do
concurrent` construct to offload to NVIDIA GPU accelerators. 

It provides a simple way of using accelerators without any extra
libraries nor deviation from the standard language. No compiler
directives or any other kind of libraries are needed. By using plain
standard Fortran the portability is not an issue. The code will use whatever
means of parallel execution available on the current platform, it might be multicore
or in this example offloading to highly parallel GPU execution. 

The 2008 standard will be supported in the foreseeable future, just like many compiler
still support the Fortran 66 standard.

This approach provides a simple, user friendly, future proof and portable approach to
offloading to accelerators. 


## Example code using SAXPY

Writing the actual Fortran 2008 standard code is surprisingly easy. Here is a
simple example using SAXPY (Single precision Z = A*X + Y).

There are two Fortran approaches one using indexing addressing
element by element :
```fortran
do concurrent (i = 1:n)
   y(i) = y(i) + a*x(i)
end do
```
or vector syntax introduced in the Fortran 90 standard:
```fortran
do concurrent (i = 1:1)
   y = y + a*x
end do
```

The vector syntax does not actually need a loop, but in order to use the
parallel `do concurrent` it needs to have a loop, but in this usage only a
single pass.

The parallel loop can be compiled for a threaded multicore architecture using:
```console
$ nvfortran -o saxpy.x  -stdpar=multicore saxpy.f90
```
or for GPU offload by using:
```console
$ nvfortran -o saxpy.x  -stdpar=gpu saxpy.f90
```

As SAXPY is mostly data movement and little computation the gain in using GPU
is small as copying of data from main memory to device memory is a limiting
factor.


## Example with more computation

Here we use an example with a bit more computation.

Using indexed syntax:

```fortran
do concurrent (i=1:M, j=1:M, k=1:M)
   Z(k,j,i)=X(k,j,i) * Z(k,j,i)**2.01_real64
   Z(k,j,i)=sin(log10(X(k,j,i) / Z(k,j,i)))
end do
```

or Fortran 90 vector syntax, where the loop has only one iteration:
```fortran
do concurrent (i=1:1)
   Z = X * Z**2.01_real64
   Z = sin(log10(X / Z))
end do
```

|   Run                      | Run time [seconds] |
|----------------------------|--------------------|
| Indexed syntax CPU  1 core | 14.5285            |
| Vector syntax  CPU  1 core | 14.5234            |
| Indexed syntax GPU  A100   |  0.4218            |
| Vector syntax  GPU  A100   |  0.4149            |

With more flops per byte transferred the speedup by offloading to
GPU is higher. A speedup of 34 compared to a single core is nice.

The NVfortran compiler is capable of generating code
to offload both using the index addressing syntax as well as
the vector syntax.


## Old legacy code example

We can also look at a matrix-matrix multiplication reference
implementation (DGEMM) code from 8-February-1989. This is found at:
[Basic Linear Algebra, level 3 matrix/matrix operations](http://www.netlib.org/blas/index.html#_level_3) or download the
[Fortran 77 reference implementation](http://www.netlib.org/blas/blas.tgz)
which contains DGEMM and also contain support files needed.

Download the legacy code, change the comment character to fit
Fortran 90 standard:
```console
$ cat dgemm.f | sed s/^*/\!/>dgemm.f90
```

The BLAS routines multiplication comes in 4 flavors:
- S single (32 bit) precision
- D double (64 bit) precision
- C complex single precision
- Z complex double precision


Assume well behaved matrices `C := alpha*A*B + beta*C` and a call to dgemm like: 
`call dgemm('n', 'n', N, N, N, alpha, a, N, b, N, beta, c, N)`

Locate the line below the line highlighted above, about line 228.
Change :
```fortran
DO 90 J = 1,N
```
with:
```fortran
DO concurrent (J = 1 : N)
```
and change the
```fortran
90 CONTINUE
```
with
```fortran
end do
```

This is all that is needed to use GPU for offloading, the rest is up to the
compiler.

Building this code needs some extra files `lsame.f` and `xerrbla.c`.
One example of how to build a threaded version for a multicore CPU could look
like:
```console
$ nvfortran -O3 -stdpar=multicore dgemm-test.f90 dgemm.f90 xerrbla.o lsame.f
```
or to build an offloaded GPU version:
```console
$ nvfortran -O3 -stdpar=gpu dgemm-test.f90 dgemm.f90 xerrbla.o lsame.f
```

| Run             | Build flags           | Cores | Performance     |
| --------------- | ----------------------|-------|-----------------|
| Reference f77   | -O3                   |   1   |   4.41 Gflops/s |
| Reference f90   | -O3 -stdpar=multicore |   2   |   6.27 Gflops/s |
| Reference f90   | -O3 -stdpar=multicore |  16   |  24.67 Gflops/s |
| Reference f90   | -O3 -stdpar=gpu       |RTX2080|  43.81 Gflops/s |
| Reference f90   | -O3 -stdpar=gpu       |  A100 | 112.23 Gflops/s |

The results are stunning: changing only one line in the old legacy
code from `do` to `do concurrent` can speed up from 4 Gflops/s to 112
Gflops/s a 25x increase in performance.

An intersting test is to compare this more then 30 year old reference code 
with a call to a modern library, the syntax is still the same. 
The scientific application fortran code will probably behave like the 30 year old example 
while libraries generally show far higher performance.  
