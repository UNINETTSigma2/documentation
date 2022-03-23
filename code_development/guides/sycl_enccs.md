---
orphan: true
---

# ENCCS SYCL workshop

The EuroCC National Competence Centre Sweden (ENCCS) has prepared course material for an
introductory workshop on SYCL spanning three half days. The course page can be found
[here](https://enccs.github.io/sycl-workshop), with the course material publicly available
[here](https://github.com/ENCCS/sycl-workshop). In the following we will demonstrate how
to compile and run the example code on Saga.

## Step 1: Load necessary modules

In order to compile the examples we will need CMake as well as a SYCL implementation that is compatible
with the Nvidia GPUs that we have available. On Saga we have hipSYCL and CMake installed globally, and
we will choose a CMake version that is compatible with the GCC toolchain that the hipSYCL module is
based upon, in this case `GCCcore/11.2.0`:

```console
[me@login-1.SAGA ~]$ module load hipSYCL/0.9.2-GCC-11.2.0-CUDA-11.4.1
[me@login-1.SAGA ~]$ module load CMake/3.22.1-GCCcore-11.2.0
```

## Step 2: Download course material

The course material can be downloaded from Github with the following command

```console
[me@login-1.SAGA ~]$ git clone https://github.com/ENCCS/sycl-workshop.git
[me@login-1.SAGA ~]$ cd sycl-workshop
[me@login-1.SAGA ~/sycl-workshop]$ ls
content  LICENSE  LICENSE.code  make.bat  Makefile  README.md  requirements.txt
```

You will here find the lesson material under `content/` in the form of rst files
(best viewed through the official [web page](https://enccs.github.io/sycl-workshop).
The code exercises are located under `content/code/`, where you will find separate
folders for each day of the course, as well as a folder with useful code snippets.

## Step 3: Configure with CMake

When we configure the build we need to tell CMake which SYCL implementation we are
going to use (hipSYCL) and which target architecture we want to compile for; `omp`
for CPU targets and `cuda:sm_60` for the Nvidia P100 GPUs we have on Saga (`cuda:sm_80`
for Betzy's A100 GPUs). We need to create separate `build` directories for each of the
examples, here for the very first `00_hello`:

```console
[me@login-1.SAGA ~/sycl-workshop]$ cd content/code/day-1/00_hello
[me@login-1.SAGA ~/sycl-workshop/content/code/day-1/00_hello]$ cmake -S . -B build -DHIPSYCL_TARGETS="omp;cuda:sm_60"
-- The CXX compiler identification is GNU 11.2.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /cluster/software/GCCcore/11.2.0/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Configuring done
-- Generating done
-- Build files have been written to: $HOME/sycl-workshop/content/code/day-1/00_hello/build
```

### Step 4: Compile and run exercises

The tutorial is organized such that you are expected to write your solution in the
`<topic>.cpp` template file of each exercise. Once you have configured a `build`
directory for a particular exercise, simply run `make` in the `build` directory:

```console
[me@login-1.SAGA ~/sycl-workshop/content/code/day-1/00_hello]$ cd build
[me@login-1.SAGA ~/sycl-workshop/content/code/day-1/00_hello/build]$ make
[ 50%] Building CXX object CMakeFiles/hello.dir/hello.cpp.o
clang-13: warning: Unknown CUDA version. cuda.h: CUDA_VERSION=11040. Assuming the latest supported version 10.1 [-Wunknown-cuda-version]
[100%] Linking CXX executable hello
[100%] Built target hello
```

Please ignore the CUDA version warning form `clang-13`, it does not seem to make a
difference. This will build an executable with the same name as the exercise (`<topic>`)
which can be launched with:

```console
[me@login-1.SAGA ~/sycl-workshop/content/code/day-1/00_hello/build]$ ./hello
[hipSYCL Warning] backend_loader: Could not load backend plugin: /cluster/software/hipSYCL/0.9.2-GCC-11.2.0-CUDA-11.4.1/bin/../lib/hipSYCL/librt-backend-cuda.so
[hipSYCL Warning] libcuda.so.1: cannot open shared object file: No such file or directory
Running on: hipSYCL OpenMP host device
Hello, world!	I'm sorry, Dave. I'm afraid I can't do that. - HAL
```

Don't mind the `[hipSYCL Warning]`, they appear since we are launching a GPU application
on the login node, which does not have the appropriate hardware drivers. The code is
still able to run, though, as we can see from the last two lines of output. This is
because we have compiled a fallback option that runs on CPU (`OpenMP host device`).

## Step 5: Run exercises on GPU nodes

In order to run the code on accelerators we to be granted GPU resources through Slurm.
We will here use an interactive session, where we get a login prompt on the GPU node,
which we can launch our applications

```console
[me@login-1.SAGA ~]$ salloc --account=<my-account> --time=1:00:00 --ntasks=1 --gpus=1 --partition=accel --mem=1G
salloc: Pending job allocation 5353133
salloc: job 5353133 queued and waiting for resources
salloc: job 5353133 has been allocated resources
salloc: Granted job allocation 5353133
salloc: Waiting for resource configuration
salloc: Nodes c7-2 are ready for job
[me@c7-2.SAGA ~]$ cd sycl-workshop/content/code/day-1/00_hello/build
[me@c7-2.SAGA ~/sycl-workshop/content/code/day-1/00_hello/build]$ ./hello
Running on: Tesla P100-PCIE-16GB
Hello, world!	I'm sorry, Dave. I'm afraid I can't do that. - HAL
```

We see that the `[hipSYCL Warning]` is gone, since we now have the GPU drivers and
libraries available. We also see that the application is able to pick up the correct
hardware, which on Saga is Tesla P100 cards, and the program output is still the same,
indicating that the code was executed correctly also on the GPU.

## Step 6: Compile and run solutions

If you get stuck at some point there is also a suggested solution to each of the
exercises, located in the `solution/` folder within each exercise. There are now two
ways to build the solution code: either copy the solution file directly to replace
the exercise template file and compile as before, or create another `build` directory
under `solution/` following the above steps to configure, build and run.

## Step 7: Play with the examples!
