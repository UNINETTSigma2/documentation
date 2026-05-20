---
orphan: true
---

# SYCL Academy tutorial

Codeplay provides a nice introductory tutorial of the basic features of SYCL in their
[SYCL Academy](https://github.com/codeplaysoftware/syclacademy)
repository. This is the course material for the standard SYCL tutorial given by Codeplay on relevant
conferences and workshops throughout the year. In the following we will demonstrate how to compile
and run the example code on Saga.

## Step 1: Load necessary modules

In order to compile the examples we will need CMake as well as a SYCL implementation that is compatible
with the Nvidia GPUs that we have available. On Saga we have hipSYCL and CMake installed globally, and
we will choose a CMake version that is compatible with the GCC toolchain that the hipSYCL module is
based upon, in this case `GCCcore/11.2.0`:

```console
[me@login-1.SAGA ~]$ module load hipSYCL/0.9.2-GCC-11.2.0-CUDA-11.4.1
[me@login-1.SAGA ~]$ module load CMake/3.21.1-GCCcore-11.2.0
```

```{note}
Some of the examples in this tutorial does not compile with the 0.9.1 version of `hipSYCL`,
so make sure to use at least version number 0.9.2.
```

## Step 2: Download course material

The course material can be downloaded from Github with the following command (remember the `--recursive` option):

```console
[me@login-1.SAGA ~]$ git clone --recursive https://github.com/codeplaysoftware/syclacademy.git
[me@login-1.SAGA ~]$ cd syclacademy
[me@login-1.SAGA ~/syclacademy]$ ls
CMakeLists.txt  CODE_OF_CONDUCT.md  External          LICENSE.md  sycl_academy.png
Code_Exercises  CONTRIBUTING        Lesson_Materials  README.md   Utilities
```

You will here find `Lesson_Materials` in the form of html slides which you can view in your browser
(best done on your local machine), as well as corresponding `Code_Exercises` for each of the lectures.
You can also follow the exercises by browsing the
[Github page](https://github.com/codeplaysoftware/syclacademy/tree/main/Code_Exercises),
where you will find explanation of the exercises in the `doc.md` file of each folder.

## Step 3: Configure with CMake

When we configure the build we need to tell CMake which SYCL implementation we are going to use (hipSYCL)
and which target architecture we want to compile for; `omp` for CPU targets and `cuda:sm_60` for the
Nvidia P100 GPUs we have on Saga (`cuda:sm_80` for the A100 cards on Betzy):

```console
[me@login-1.SAGA ~/syclacademy]$ mkdir build
[me@login-1.SAGA ~/syclacademy]$ cd build
[me@login-1.SAGA ~/syclacademy/build]$ cmake -DSYCL_ACADEMY_USE_HIPSYCL=ON -DHIPSYCL_TARGETS="omp;cuda:sm_60" ..
```

Hopefully no error occurred on this step.

```{tip}
If you got a `syclacademy/External/Catch2 does not contain a CMakeLists.txt file` error you may
have forgotten to download the submodules of the git repo (`--recursive` option in the clone).
```


## Step 4: Compile and run exercises

The tutorial is organized such that you are expected to write your solution code in the `source.cpp` file
of each exercise in the `Code_Exercises` folder based on the text given in the corresponding `doc.md` file.
You can then compile your source file for exercise 1 with the following command:

```console
[me@login-1.SAGA ~/syclacademy/build]$ make exercise_01_compiling_with_sycl_source
```

after which the resulting executable (with the same long and cumbersome name as the build target) can be found under:

```console
[me@login-1.SAGA ~/syclacademy/build]$ ls Code_Exercises/Exercise_01_Compiling_with_SYCL/
CMakeFiles           CTestTestfile.cmake                     Makefile
cmake_install.cmake  exercise_01_compiling_with_sycl_source
```

and you can execute you program with

```console
[me@login-1.SAGA ~/syclacademy/build]$ Code_Exercises/Exercise_01_Compiling_with_SYCL/exercise_01_compiling_with_sycl_source
===============================================================================
All tests passed (1 assertion in 1 test case)
```

If it shows that the test passes it means that your code did not crash, which is good news.

## Step 5: Compile and run solutions

If you get stuck at some point there is also a suggested solution to each of the exercises, called `solution.cpp` which you
can compile using the same long and cumbersome exercise name as before, but with `_solution` instead of `_source` in the end

```console
[me@login-1.SAGA ~/syclacademy/build]$ make exercise_01_compiling_with_sycl_solution
[me@login-1.SAGA ~/syclacademy/build]$ Code_Exercises/Exercise_01_Compiling_with_SYCL/exercise_01_compiling_with_sycl_solution
===============================================================================
All tests passed (1 assertion in 1 test case)
```

## Step 6: Run exercises on GPU nodes

The main point of all these exercises is of course to run the code on accelerators, and in order to do that
you need to ask for GPU resources through Slurm, here as an interactive job on the `accel` partition:

```console
[me@login-1.SAGA ~/syclacademy/build]$ salloc --account=<my-account> --time=1:00:00 --ntasks=1 --gpus=1 --partition=accel --mem=1G
salloc: Pending job allocation 5353133
salloc: job 5353133 queued and waiting for resources
salloc: job 5353133 has been allocated resources
salloc: Granted job allocation 5353133
salloc: Waiting for resource configuration
salloc: Nodes c7-8 are ready for job
[me@c7-8.SAGA ~/syclacademy/build]$ Code_Exercises/Exercise_01_Compiling_with_SYCL/exercise_01_compiling_with_sycl_solution
===============================================================================
All tests passed (1 assertion in 1 test case)
```

The test still passed on the GPU node, yay!

## Step 7: Play with the examples!
