# Getting started Betzy

This page is a quick intro for the projects invited to participate in the pilot
phase and relevant information will be eventually merged into main
documentation.

## Getting access

### Access during pilot

During the pilot phase, access to Betzy is allowed only from the UH-sector.

You may login either by connecting via VPN or SSH tunnel to the university 
network and from there to Betzy via SSH, or from one of the other HPC machines 
(Fram or Saga), or NIRD.

Please note that you may need to [reset your password](https://www.metacenter.no/user/).
For more information, check out the [password policy
news](https://www.sigma2.no/sigma2-launches-new-password-policy).

The address for Betzy login nodes is `login.betzy.sigma2.no`.

### Hardware

Each of the compute nodes have *128 cores* (with logical cores it looks like 256). There are two sockets
each with a 64 core AMD processor. The AMD processors are not recognied by the Intel compiler and 
this may cause issues (se later). The AMD processors support 256 bits Vector instruction, *AVX* and *AVX2*.
Each compute node has *256 GiB* of memory split up into 8 NUMA nodes (numactl -H).

### Support

Please use the regular [support channel](/getting_help/support_line.md) to submit requests 
and mark the case with *Betzy*.

## Queue System

Betzy uses the SLURM queuing system. For more information about the queue
system on Betzy, please check out the [job types](/jobs/choosing_job_types.md)
page.

**NOTE**: During the pilot phase there will be no CPU core limitations imposed. 

For generic information about job submission, please check out our documentation [here](/jobs/submitting.md).


## Storage

Betzy uses a DDN powered 2.5 PB Lustre parallel file system with 51 GB/s bandwidth and 500k+  metadata operations per second.
The Lustre file system is mounted under `/cluster`.

```{warning}
**Betzy: NO BACKUP**

Please note that **NO BACKUP** is taken during the pilot phase.

The `/cluster` file system might be reformatted at the end of the pilot phase,
before machine being placed into production.
```

For additional information about the file system configuration, follow this [link](/files_storage/clusters.md).
File system performance tuning and best practices is listed
[here](/files_storage/performance/lustre.md). 

NIRD project file systems - `/trd-project[1-4]` - will be mounted on the Betzy
login nodes, similarly to Fram and Saga HPC clusters.

## Software

As on Fram and Saga, scientific software on Betzy will be installed using the EasyBuild system, and the Lmod modules tool
will be used for changing environment setup via modulefiles.

The two *common toolchains* `foss` and `intel` will be installed on Betzy.

### foss toolchain
* GCC compilers (`gcc`, `g++`, `gfortran`)
* Open MPI library
* OpenBLAS (including LAPACK) + ScaLAPACK
* FFTW library

### intel toolchain
* Intel compilers (`icc`, `icpc`, `ifort`)
* Intel MPI library
* Intel MKL library (including BLAS, LAPACK, ScaLAPACK, FFT)

Regarding compiler optimization this needs to be investigated case by case. Aggressive optimization should be added only to files
where it makes a difference, as it increases the probability of the compiler generating wrong code or exposing
floating-point issues in the application. A few starting suggestions are:

### Compiler flags
A set of suggested compiler flags are given below, these have been tested and used, but users are 
advised to read the documentation and try other combinations as not all code are alike.

#### gcc/gfortran
*  -O3
*  -O3 -march=znver2 -mtune=znver2 (recommended)
*  -O3 -march=znver2 -mtune=znver2 -mfma -mavx2 -m3dnow -fomit-frame-pointer

#### ifort/icc/ipcp 
* -O3  
* -O3 -march=core-avx2 
* -O3 -xavx
* -O3 -xavx2 (recommended, for all execpt for main() )
* -O3 -xcore-avx2 

The above applies to GCC 9.3 and Intel 2019b, and the choices are listed in increased performance as obtained with a DGEMM matrix
multiplication test (which show significant performance improvement), it is also verified using one of the major benchmarks used. 
Please notice that ifort performs substantially better (2-4x) than gfortran with the dgemm.f test. Also, building the main routine in the program 
file with *-xcore-avx2*, *-xavx* or *-xavx2* is not recommended. 
It's known that building the main() (C and Fortran) with these flags trigger the Intel processor run time check, causing the application to abort.


### MPI libraries

Both OpenMPI and Intel MPI are installed. Built both for GNU and Intel. Experience have shown that performance varies. Both OpenMPI and Intel MPI are
supported. OpenMPI is built with support for GNU compiler and Intel.  

#### OpenMPI
* mpicc
* mpicxx
* mpif90

For running with OpenMPI the processor binding is beneficial, adding *-bind-to core* is generally a good idea.

#### Intel MPI
* mpiicc
* mpiicpc
* mpiifort

For running Intel MPI, the settings
* I_MPI_PIN=1 
* I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255 (run only on physical cores)

are a good starting points. 

For a correct SLURM job script file the only command needed to launch MPI programs are for :
* OpenMPI : mpirun -bind-to core ./a.out
* Intel MPI: mpirun ./a.out


#### Running Hybrid models

For hybrid models it's important to set up SLURM to provide access to all vailable cores. An example could look like this:

```
#SBATCH --ntasks=2048
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
```

This will request 32 MPI ranks per node, and leave cores for 4 threads per rank, e.g. each of the 32 ranks can shedule 4 threads
yielding a total of 128 cores which is the maximum number of cores on each compute node.  The *exclusive* is important, if not set
SLURM will only allow the 32 cores allocated to be used (this will place all 4 threads onto one core). In order to have free access
to all cores the *exclusive* need to be set. 


### MKL library

The MKL perform run time check to select correct code to invoke. This test fail to find any Intel processor and hence select a code compatible with
all x86-64 processors. Setting the environment flag *MKL_DEBUG_CPU_TYPE=5* will force MKL to select code that uses *AVX2* instructions,
hence increase performance significantly. 

It is possible to use Intel MKL library with the GNU compilers, this require some work resolve the symbol names at link time and library 
path at run time, but can provide a nice performance boost to both linear algebra and Fourier transforms.


