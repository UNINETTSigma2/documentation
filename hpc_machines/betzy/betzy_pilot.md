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

<div class="alert alert-warning">
  <h4>Betzy: NO BACKUP</h4>
  <p>
    Please note that <strong>NO BACKUP</strong> is taken during the pilot phase.
	</p>
	<p>
    The `/cluster` file system might be reformatted at the end of the pilot phase, before machine being placed into production.
  </p>
</div>

For additional information about the file system configuration, follow this [link](/files_storage/clusters.md).
File system performance tuning and best practices is listed
[here](/files_storage/performance/lustre.md). 

NIRD project file systems - `/trd-project[1-4]` - will be mounted on the Betzy
login nodes, similarly to Fram and Saga HPC clusters.

## Software

As on Fram and Saga, scientific software on Betzy will be installed using the EasyBuild system, and the Lmod modules tool
will be used for changing environment setup via modulefiles.

The two *common toolchains* `foss` and `intel` will be installed on Betzy.

`foss` toolchain
* GCC compilers (`gcc`, `g++`, `gfortran`)
* Open MPI library
* OpenBLAS (including LAPACK) + ScaLAPACK
* FFTW library

`intel` toolchain
* Intel compilers (`icc`, `icpc`, `ifort`)
* Intel MPI library
* Intel MKL library (including BLAS, LAPACK, ScaLAPACK, FFT)

Regarding compiler optimization this needs to be investigated case by case. Aggressive optimization should be added only to files
where it makes a difference, as it increases the probability of the compiler generating wrong code or exposing
floating-point issues in the application. A few starting suggestions are:
* gcc/gfortran : -O3, -O3 -march=znver2 -mtune=znver2 or -O3 -march=znver2 -mtune=znver2 -mfma -mavx2 -m3dnow -fomit-frame-pointer
* ifort/icc    : -O3, -O3 -march=core-avx2, -O3 -xavx, -O3 -xavx2, -O3 -xcore-avx2 

The above applies to GCC 9.3 and Intel 2019b, and the choices are listed in increased performance as obtained with a DGEMM matrix
multiplication test, it is also verified using one of the major benchmarks used. Please notice that ifort performs substantially better than 
gfortran with the dgemm.f test. Also, building the main routine in the program file with -xcore-avx2, -xavx or -xavx2 is not recommended. 
It's known that building the main() (C and Fortran) with these flags trigger the Intel processor run time check, causing the application to abort.

'MPI libraries'
Both OpenMPI and Intel MPI are installed. Built both for GNU and Intel. Experience have shown that performance varies. For large applications
Intel MPI has provided the best results so far. For Intel MPI Setting the I_MPI_PIN=1 and I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255 are a good 
starting point.

'MKL library'
The MKL perform run time check to select correct code to invoke. This test fail to find any Intel processor and hence select a code compatible with
all x86-64 processors. Setting the environment flag MKL_DEBUG_CPU_TYPE=5 will force MKL to select code that uses AVX2 instructions,
hence increase performance significantly. 

