# Calling fortran routines from Python 

## Introduction
While Python an effective language for development it is not very fast
at executing code. There are several tricks available to get high 
numerical performance of which calling fortran routines is one.

While libraries functions in both numpy and scipy perform nicely in
many cases, one often need to write routines for which no library
exist. Either writing from scratch or use fortran routines from
co-workers or other sources. In any case it's a good way of getting
high performance for time consuming part of the run.

Below is covered usage of :
- Plain fortran with GNU fortran (the default)
- Fortran with calls to math library (MKL) 
- The Intel fortran compiler to compile your fortran source code
- Optimising performance for fortran code, compiler flags.
- Intel fortran and MKL 
- Intel, fortran and multithreaded MKL
- Python with multithreaded OpenMP fortran routines

A short disclaimer With regards to matrix matrix multiplication the library 
in numpy is comparable in performance to the Intel MKL. 

Another disclaimer is that this have been tested on Saga. There might some
minor issues on Betzy with AMD processors, not having 512 bits avx.

## Using the numpy interface
The package [numpy](https://numpy.org/) contains tools to facilitate
calling fortran routines directly from Python. The utility f2py3 can
be used or more indirectly by launching Python with a module and
processing the fortran source code. In both cases the fortran code
containing definitions of subroutines will be compiled using a fortran
compiler into object files which subsequently are linked into a
single shared object library file (an .so file). 

A nice introduction by NTNU is
[available](https://www.numfys.net/howto/F2PY/). It cover some basics
and should be read as an introduction. Issues with arrays arguments
and assumed shapes are explained.

Modern fortran uses «magic» constants (they can any number, but often 
they are  equal to the number of bytes, but not always, don't rely on this) 
to set the attributes like size or range of variables. Normally specified in 
the number of bits for a given variable. This can be done using self 
specified ranges with the help of the `kind` function.
```fortran
subroutine foo
	implicit none
	int32 = selected_int_kind(8)
	int64 = selected_int_kind(16)
	real32  = selected_real_kind(p=6,r=20)
	real64  = selected_real_kind(p=15,r=307)
	
	integer(int32) :: int
	integer(int64) :: longint
```
or a simpler solution is to use a standard fortran module:
```fortran
subroutine foo
	use iso_fortran_env
	implicit none
	
	real(real32) :: float
	real(real64) :: longfloat
```	
While the first one is more pedagogic, the second one is simpler and 
 [iso_fortran_env](https://fortranwiki.org/fortran/show/iso_fortran_env)
contain a lot more information.

Python support both 32 and 64 bit integers and floats. However, the mapping
between fortran specification and Python/Numpy it not set by default.
In order to map from fortran standard naming to C naming map need to be 
provided. The map file need to reside in the working directory and must 
have the name `.f2py_f2cmap`. An example mapping fortran syntax to C syntax
for simple integers and floats can look like :
```python
dict(real=dict(real64='double', real32='float'),
	 complex=dict(real32='complex_float', real64='complex_double'),
     integer=dict(int32='int', int64='long')
	 )
```
This helps the f2py3 to map the fortran data types into the 
corresponding C data types. Alternative is to use [C mapping directly](https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fC_005fBINDING.html#ISO_005fC_005fBINDING).

For complex variables the same logic applies, the size is measured in bits to fit
two numbers (real and imaginary parts) occupying 64 bits each, hence 128 bits.
```python
x=np.zeros((n), dtype=np.complex128, order='F')
y=np.zeros((n), dtype=np.complex128, order='F')
```
and corresponding fortran code, there each number is specified as 64 bits each:
```fortran
complex(real64), dimension(n), intent(in) :: x
complex(real64), dimension(n), intent(inout):: y
```

The importance of keeping control over data types and their ranges cannot
be stressed more than pointing to [Ariane-5 failure](https://en.wikipedia.org/wiki/Ariane_flight_V88) or even worse, killing people, the 
[Therac-25](https://en.wikipedia.org/wiki/Therac-25) incident.

### compiling fortran code
To start using Python with fortran code a module need to be loaded,
`module load  Python/3.9.6-GCCcore-11.2.0`

The command line to generate the Python importable module can be one of the 
following, with the second could be used if f2py3 is not available.
- `f2py3 -c pi.f90 -m pi`
- `python3 -m numpy.f2py -c pi.f90 -m pi`
In both cases a module will be generated which could be imported as a 
normal Python module. The `-m pi` is the given name for the module, here it's
identical to the name of the subroutine, but don't need to be.

A simple fortran routine to calculate Pi :
```fortran
subroutine pi(p,n)
  use iso_fortran_env
  implicit none
  real(real64), intent(out) :: p
  integer(int64), intent(in) :: n
  
  integer(int64) :: j
  real(real64) :: h, x, sum
  
  sum=0.0_real64 ! set accumulating vars to 0.
  h = 1.0_real64/n
  do j = 1,n
      x = h*(j-0.5_real64)
      sum = sum + (4.0_real64/(1.0_real64+x*x))
   end do
   p = h*sum
   return
 end subroutine pi
```
Be aware that intention of parameters is important. Also that variables are 
not initiated during repeated calls, hence set accumulating variables to zero 
in the body, not during declaration . Once the routine is loaded into memory 
the variables reside in memory. There is no magic initialisation for each 
subsequent call (look into the  [save statement](https://stackoverflow.com/questions/2893097/fortran-save-statement) in fortran).

This fortran routine can be called from a Python script like:
 ```python
import pi

p=pi.pi(1000)

print("Pi calculated ",p)
 ```
With a result like:
 ```
Pi calculated  3.1415927369231227
```
We import the module generated, the name is pi which correspond to the last 
`-m <name>` argument, while the function call to `pi` is the same name as 
the fortran routine. 

### Performance issues
While Python is easy to write and has many very nice features and applications,
numerical performance is not among them.

It the following examples matrix matrix multiplication is used as an
example, this is a well known routine making it a good candidate for
performance comparison.


The following code is used to illustrate the performance using Python:
```python
print("Matrix multiplication example")
x=np.zeros((n, n), dtype=np.float64, order='F')
y=np.zeros((n, n), dtype=np.float64, order='F')
z=np.zeros((n, n), dtype=np.float64, order='F')
x.fill(1.1)
y.fill(2.2)

start = time.perf_counter()
for j in range(n):
    for l in range(n):
        for i in range(n):
            z[i,j] = z[i,j] + x[i,l]*y[l,j]
print(f"Python code {time.perf_counter() - start:2.4f} secs")
print(z)
```


The following fortran code is used for matrix matrix multiplication:
```fortran
subroutine mxm(a,b,c,n) 
  implicit none
  integer, parameter :: real64  = selected_real_kind(p=15,r=307)
  integer, parameter :: int32 = selected_int_kind(8)

  real(real64), dimension(n,n), intent(in)  :: a,b
  real(real64), dimension(n,n), intent(inout) :: c
  integer(int32), intent(in)  :: n
  integer(int32) :: i,j,l

  do j = 1,n
     do l = 1,n
        do i = 1,n
           c(i,j) = c(i,j) + a(i,l)*b(l,j)
        enddo
     enddo
  enddo
  
end subroutine mxm
```
Comparing Python with fortran using the following commands:
```
f2py3 --opt="-Ofast -fomit-frame-pointer -march=skylake-avx512"  -c mxm.f90 -m mxm
```
and running the Python script 
`python3 mxm.py`

The Python script used to call the fortran code is:

```python
a=np.zeros((n, n), dtype=np.float64, order='F')
b=np.zeros((n, n), dtype=np.float64, order='F')
c=np.zeros((n, n), dtype=np.float64, order='F')
a.fill(1.1)
b.fill(2.2)
start = time.perf_counter()
mxm.mxm(a,b,c,n)
print(f"f90 mxm {time.perf_counter() - start:2.4f} secs")
```

The results are staggering, for the matrix matrix multiplication the simple 
fortran implementation perform over 2000 times faster than the fortran code.

| Language  | Run time in seconds |
|-----------|---------------------|
| Python    | 757.2706            |
| f90       | 0.3099              |

This expected as the compiled fortran code is quite efficient while Python 
is interpreted. 


### Using libraries, MKL
The Intel Math Kernel Library is assumed to be well known for its
performance.  It contains routines that, in most cases, exhibit very
high performance. The routines  are also for the most part threaded to 
take advantage of multiple cores.

In addition to the module already loaded 
`module load  Python/3.9.6-GCCcore-11.2.0`
one more module is needed to use Intel MKL:
`module load imkl/2022.2.1`
(This module set many environment variables, we use `$MKLROOT` to
set the correct path for MKL library files.)

As f2py3 is a wrapper some extra information is needed to link with
the MKL libraries. The simplest is to use static linking:
```bash
f2py3 --opt="-Ofast -fomit-frame-pointer -march=skylake-avx512"\
 ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a\
 ${MKLROOT}/lib/intel64/libmkl_sequential.a\
 ${MKLROOT}/lib/intel64/libmkl_core.a\
 -c mxm.f90 -m mxm 
``` 
The above commands link in the `dgemm` routine from MKL. 
```fortran
subroutine mlib(c,a,b,n) 
  implicit none
  integer, parameter :: real32  = selected_real_kind(p=6,r=20)
  integer, parameter :: real64  = selected_real_kind(p=15,r=307)
  integer, parameter :: int32 = selected_int_kind(8)
  integer, parameter :: int64 = selected_int_kind(16)

  real(real64), dimension(n,n), intent(in)  :: a,b
  real(real64), dimension(n,n), intent(out) :: c  
  integer(int32), intent(in)  :: n
  real(real64) :: alpha=1.0_real64, beta=1.0_real64
  
  call dgemm('n', 'n', n, n, n, alpha, a, n, b, n, beta, c, n)

end subroutine mlib
```
and a Python script to call it :
```python
a=np.zeros((n, n), dtype=np.float64, order='F')
b=np.zeros((n, n), dtype=np.float64, order='F')
c=np.zeros((n, n), dtype=np.float64, order='F')
a.fill(1.1)
b.fill(2.2)
c=np.zeros((n, n), dtype=float64, order='F')
start = time.perf_counter()
mxm.mlib(a,b,c,n)
print(f"mxm MKL lib {time.perf_counter() - start:2.4f} secs")
```
Running the Python script with n=5000 we get the results below.

| Routine      | Run time in seconds | 
|--------------|---------------------|
| Fortran code |  88.566             |
| MKL library  |  2.90               |


### Using different fortran compiler, intel
While the gfortran used by default generate nice executable code it does not
always match the intel fortran compiler when it comes to performance. 
It might be beneficial to switch to the intel compiler.

In order to have Python, Intel compiler and MKL together load the module:
`SciPy-bundle/2022.05-intel-2022a`

Then we build compile the fortran code,
```bash
f2py3  --fcompiler=intelem  --opt="-O3 -xcore-avx512"\
 -c mxm.f90 -m mxm
```
Running the same Python script with n=5000 we arrive at the following 
run times:

| Compiler/library  | Run times seconds |
|-------------------|-------------------|
| GNU fortran       |  88.566           |
| Intel ifort       |   9.5695          |

The Intel compiler is known for its performance when compiling the
matrix matrix multiplication.

We can also use the MKL library on conjunction with the Intel
compiler, but it's a bit more work. First static linking:

```bash
f2py3  --fcompiler=intelem --opt="-O3 -xcore-avx512"\
 ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a\
 ${MKLROOT}/lib/intel64/libmkl_sequential.a\
 ${MKLROOT}/lib/intel64/libmkl_core.a\
 -c mxm.f90 -m mxm
```

| Compiler/library  | Run times seconds |
|-------------------|-------------------|
| GNU fortran       |  88.566           |
| Intel ifort       |   9.5695          |
| MKL dgemm         |   2.712           |

It's also possible to use dynamic linking,
```bash
f2py3  --fcompiler=intelem --opt="-O3 -xcore-avx512"\
 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_avx512\
 -c mxm.f90 -m mxm
 ```
 Then it's just to launch as before. Performance is comparable as it's the
 same library. 
 
Testing for even higher performance using the Intel compiler `ifort`
we can try more optimising flags:

| ifort flags                               | Run time    |
|-------------------------------------------|-------------|
| Defaults (no flags given)                 | 1122  secs. |
| -O2                                       | 1110 secs.  |
| -O3                                       | 153 secs.   |
| -O3 -xavx2                                | 81.8 secs.   |
| -O3 -xcore-avx512                         | 72.5 secs.  |
| -O3 -xcore-avx512 -qopt-zmm-usage=high    | 54.1 secs.  |
| -Ofast -xcore-avx512 -qopt-zmm-usage=high | 53.9 secs.  |
| -Ofast -unroll -xcore-avx512 -qopt-zmm-usage=high -heap-arrays -fno-alias | 53.7 secs. |
| -fast -unroll -xcore-avx512 -qopt-zmm-usage=high | 53.6 secs. |

Selecting the _right_ flags can have dramatic affect on performance. Adding to this what's 
optimal flag for one routine might not be right for other. 
 
### Using many cores with MKL library

As the MKL libraries are multithreaded they can be run on multiple cores.

To achieve this it just to build using multithreaded versions of the library, 
using static linking :
 ```bash
f2py3  --fcompiler=intelem --opt="-O3 -xcore-avx512"\
 ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a\
 ${MKLROOT}/lib/intel64/libmkl_intel_thread.a\
 ${MKLROOT}/lib/intel64/libmkl_core.a\
 -c mxm.f90 -m mx
 ```
or dynamic linking:
```bash
f2py3  --fcompiler=intelem --opt="-O3 -xcore-avx512"\
 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_avx512 -liomp5\
 -c mxm.f90 -m mxm
```
The OpenMP `OMP_NUM_THREADS` environment variable can the be used to 
control the number of cores to use.

This time we run the Python script with a bit larger size, n=10000,
`export OMP_NUM_THREADS=2` and larger.

| Threads | Run times in seconds |
|---------|----------------------|
|   1     |     21.2914          |
|   2     |     12.5923          |
|   4     |     7.0082           |
|   8     |     4.1504           |

While scaling is not perfect there is a significant speedup by using 
extra cores.


### Using many cores with fortran with OpenMP 
It's possible to call fortran functions with OpenMP directives
getting speedup using several cores. A nice alternative when dealing with
real world code for which no library exist.

Consider the following fortran OpenMP code:
```fortran
subroutine piomp(p, n)
  use iso_fortran_env	
  real(real64), intent(out) :: p
  integer(int64), intent(in) :: n
  integer(int64) :: i
  real(real64) ::  sum, x, h
  
  h = 1.0_real64/n
  sum = 0.0_real64
!$omp parallel do private(i) reduction(+:sum)
!This OpenMP inform the compiler to generate a multi threaded loop
  do i = 1,n
     x = h*(i-0.5_real64)
     sum = sum + (4.0_real64/(1.0_real64+x*x))
  enddo
  p = h*sum
```

Building the module for Python using :
```bash
f2py3  --fcompiler=intelem --opt="-qopenmp -O3 -xcore-avx512"\
 -D__OPENMP -liomp5  -c pi.f90 -m pi
```
The openmp library is linked explicitly `-liomp5` (for GNU it's -lgomp).

Running using the following Python script :
```python
import time
import pi

n=50000000000

start = time.perf_counter()
p=pi.pi(n)
print("Pi calculated ",p," ",time.perf_counter() - start," seconds")

start = time.perf_counter()
p=pi.piomp(n)
print("Pi calculated ",p," ",time.perf_counter() - start," seconds")
```

Scaling performance is nice:

|Cores | Run time in seconds |
|------|---------------------|
|  1   |  31.26              |
|  2   |  16.28              |
|  4   |   8.528             |
|  8   |   4.217             |
| 16   |   2.547             |
| 32   |   1.900             |




