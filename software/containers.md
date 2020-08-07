

# Running Singularity and Docker containers

It is now possible to run Singularity containers on Saga, Fram, and Betzy.

This gives the users enormous flexibility to bring a full software stack
already set up to the cluster. This can make software installations and
dependencies more reproducible and more portable across clusters.

And since Singularity can build containers from Docker images, it is also
possible to run Docker images through Singularity and we will show you how.


## Simple example to get started

First we pull a "hello world" Singularity image from https://singularity-hub.org/:
```
$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```
We do the `singularity pull` step on the login node. We cannot do this step inside a job
script because compute nodes cannot access the web.

Once we have the `hello-world.sif` file, we can test it out with the following
job script on Saga (adjust "myaccount"; on Fram/Betzy you will need to remove
the line containing `#SBATCH --mem-per-cpu=1000M` but the rest should work as
is):

```bash
#!/bin/bash

#SBATCH --account=myaccount
#SBATCH --job-name=singularity-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=00:03:00

echo
echo "check that we can read the current directory from the container:"
singularity exec hello-world.sif ls

echo
echo "what is the operationg system on the host?"
cat /etc/os-release

echo
echo "what is the operationg system on the container?"
singularity exec hello-world.sif cat /etc/os-release
```

This produces the following output. Notice how on the container we are on a
Ubuntu operating system:
```
check that we can read the current directory from the container:
hello-world.sif
run.sh
slurm-1119935.out

what is the operationg system on the host?
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

what is the operationg system on the container?
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 14.04.6 LTS"
VERSION_ID="14.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"
```


## Fetching images from the web

As mentioned above, we fetch images on the login node. We cannot do this step
inside a job script because compute nodes cannot access the web.

You can fetch an image from https://singularity-hub.org/:
```
$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```
Or from https://cloud.sylabs.io/library:
```
$ singularity pull --name alpine.sif library://alpine:latest
```
Or from https://hub.docker.com/:
```
$ singularity pull --name alpine.sif docker://alpine:latest
```
Or https://quay.io/:
```
$ singularity pull --name openmpi-i8.sif docker://quay.io/bast/openmpi-i8:4.0.4-gcc-9.3.0
```


## MPI example

Here we will try a bit more advanced example. First we will create a container
from a definition file, then we will try to run the container on multiple
nodes.


### Creating a Singularity container from a definition file

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


### Running the container on multiple nodes

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
