(using-mkl-efficiently)=

# Using MKL efficiently

The Intel Math Kernel Library is a popular library
for vector and matrix operations, solving eigenvalue problems, and much more.
On AMD processors it is important that you verify whether MKL is using
its best performing routines. You can also force it to and below we discuss
how this can be done.

MKL performs run time checks at startup to select the
appropriate **Intel** processor. When it cannot work ut the processor type a
least common instruction set is set is selected yielding lower performance.


## Mixing Intel compiler and MKL versions

Users are adviced to check if there is any performance difference between
Intel 2019b and the 2020 versions. We recommend to **not mix different compiler
versions and Math Kernel Library (MKL) versions**,
e.g. building using 2020 compilers and then linking with MKL 2019.


## MKL_DEBUG_CPU_TYPE

To instruct MKL to use a more suitable instruction set a debug variable can be
set, e.g. `export MKL_DEBUG_CPU_TYPE=5`.

However,
the `MKL_DEBUG_CPU_TYPE` environment variable does not work for Intel compiler distribution
2020 and and newer.


## Forcing MKL to use best performing routines

MKL issue a run time test to check for genuine Intel processor. If
this test fail it will select a generic x86-64 set of routines
yielding inferior performance. This is well documented
[here](https://en.wikipedia.org/wiki/Math_Kernel_Library) and
remedies are discussed in
[Intel MKL on AMD Zen](https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html).

It has been found that MKL calls a function called
`mkl_serv_intel_cpu_true()` to check the current CPU. If a genuine
Intel processor is found, it returns 1.

The trick is to
bypass this by writing a dummy function which always
returns 1 and place this first in the search path (below we show how):
```c
int mkl_serv_intel_cpu_true() {
	return 1;
}
```

Save this into a file called `trick.c` and compile it into a shared library
using the following command:
`gcc -shared -fPIC -o libtrick.so trick.c`

To put the new shared library first in the search path we can use a preload environment variable:
`export LD_PRELOAD=<path to lib>/libtrick.so`.

In addition, setting the environment variable `MKL_ENABLE_INSTRUCTIONS` to
`AVX2` can also have a significant effect on performance.  Just changing it to
`AVX` can have a significant negative impact.

Setting it to `AVX512` and launching it on AMD it does not fail, MKL probably
tests if the requested feature is available.

The following table show the recorded performance obtained with the HPL (the
top500) test using a small problem size and a single Betzy node:

| Settings                                              | Performance    |
|------------------------------------------------------:|:--------------:|
| None                                                  | 1.2858 Tflop/s |
| LD_PRELOAD=./libtrick.so                              | 2.7865 Tflop/s |
| LD_PRELOAD=./libtrick.so MKL_ENABLE_INSTRUCTIONS=AVX  | 2.0902 Tflop/s |
| LD_PRELOAD=./libtrick.so MKL_ENABLE_INSTRUCTIONS=AVX2 | 2.7946 Tflop/s |
