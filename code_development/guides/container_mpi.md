---
orphan: true
---

# Building MPI containers

The following example will demonstrate how to _build_ Singularity containers on
your laptop which are suitable for execution on our HPC systems (Saga, Fram, Betzy).
If you are only interested in _running_ existing containers,
see {ref}`Running containers <running-containers>`.

## Creating a Singularity container from a definition file

**We do this step on our own laptop/computer, not on the cluster**.
This is because Singularity needs root rights to build the container.
We will later
upload the generated container file to the cluster.

You need to have Singularity installed on your laptop for this to work
(follow e.g. https://sylabs.io/guides/3.3/user-guide/installation.html).

We start with the following definitions file (`example.def`; this is a simplified
version based on the example in https://sylabs.io/guides/3.3/user-guide/mpi.html):
```
Bootstrap: docker
From: ubuntu:latest

%environment
    export OMPI_DIR=/opt/ompi

%post
    apt-get update && apt-get install -y wget git bash gcc gfortran g++ make file
    export OMPI_DIR=/opt/ompi
    export OMPI_VERSION=4.0.1
    export OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-$OMPI_VERSION.tar.bz2"
    mkdir -p /tmp/ompi
    mkdir -p /opt
    cd /tmp/ompi && wget -O openmpi-$OMPI_VERSION.tar.bz2 $OMPI_URL && tar -xjf openmpi-$OMPI_VERSION.tar.bz2
    cd /tmp/ompi/openmpi-$OMPI_VERSION && ./configure --prefix=$OMPI_DIR && make install
```

From this we build a container (we are still on the laptop, not on the cluster):
```
$ sudo singularity build example.sif example.def
```

This takes a couple of minutes:
```
[... lots of output ...]

INFO:    Adding environment to container
INFO:    Creating SIF file...
INFO:    Build complete: example.sif
```

Once `example.sif` is generated, we can `scp`
the container file to the cluster.


## Running the container on multiple nodes

We assume that we have the container file `example.sif` from the step before on
the cluster.  We will also fetch `mpi_hello_world.c` from
https://mpitutorial.com/tutorials/mpi-hello-world/.

We are ready to test it out with the following job script on Saga (adjust
"myaccount"; on Fram/Betzy you will need to remove the line containing `#SBATCH
--mem-per-cpu=1000M` but the rest should work as is):
```bash
#!/bin/bash

#SBATCH --account=myaccount
#SBATCH --job-name=singularity-test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=00:03:00

singularity exec example.sif /opt/ompi/bin/mpirun --version
singularity exec example.sif /opt/ompi/bin/mpicc mpi_hello_world.c

module purge
module load foss/2020a

mpirun --bind-to core -n ${SLURM_NTASKS} singularity exec example.sif ./a.out
```

The interesting part of the output is:
```
mpirun (Open MPI) 4.0.1

[...]

Hello world from processor c2-4, rank 2 out of 16 processors
Hello world from processor c2-4, rank 3 out of 16 processors
Hello world from processor c2-4, rank 0 out of 16 processors
Hello world from processor c2-4, rank 1 out of 16 processors
Hello world from processor c2-21, rank 14 out of 16 processors
Hello world from processor c2-21, rank 13 out of 16 processors
Hello world from processor c2-21, rank 12 out of 16 processors
Hello world from processor c2-21, rank 15 out of 16 processors
Hello world from processor c2-18, rank 8 out of 16 processors
Hello world from processor c2-18, rank 11 out of 16 processors
Hello world from processor c2-18, rank 9 out of 16 processors
Hello world from processor c2-18, rank 10 out of 16 processors
Hello world from processor c2-11, rank 6 out of 16 processors
Hello world from processor c2-11, rank 4 out of 16 processors
Hello world from processor c2-11, rank 7 out of 16 processors
Hello world from processor c2-11, rank 5 out of 16 processors
```

Looks like it's working!
