# Getting started Betzy

This page is a quick intro for the projects invited to participate in the pilot
phase and relevant information will be eventually merged into main
documentation.

## Getting access

### Access during pilot

To be added.

### Support

Please use the regular [support channel](../help/support.md) to submit requests 
and mark the case with *Betzy*.

## Queue System

Betzy uses the SLURM queuing system. For more information the queue system on
Betzy, please check out [job types](../jobs/job_types.md) page.

**NOTE**: During the pilot phase there will be no CPU core limitations imposed. 

For generic information about job submission, please check out our documentation [here](../jobs/queue_system.md).


## Storage

Betzy uses a DDN powered 2.5 PB Lustre parallel file system with 51 GB/s bandwidth and 500k+  metadata operations per second.
The Lustre file system is mounted under `/cluster`.

<div class="alert alert-warning">
  <h4>Betzy: NO NACKUP</h4>
  <p>
    Please note that *NO BACKUP* is taken during the pilot phase.
    The `/cluster` file system might be reformatted at the end of the pilot phase, before machine being placed into production.
  </p>
</div>

For additional information about the file system configuration, follow this [link](/storage/clusters/md).
File system performance tuning and best practices is listed
[here](/storage/performance/lustre.md). 

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
