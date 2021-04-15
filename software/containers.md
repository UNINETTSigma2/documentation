(running-containers)=
# Running containers

```{note}
Currently, [Singularity](https://sylabs.io/singularity/) is the only supported container
solution on our HPC systems (Saga, Fram, Betzy). However, since Singularity can build
containers from Docker images, it is also possible to run [Docker](https://www.docker.com/)
containers through Singularity.
```

Containers give the users the flexibility to bring a full software stack to the
cluster which has already been set up, which can make software installations and
dependencies more reproducible and more portable across clusters.

Singularity is already installed globally on all our systems, and should be
immediately available on your command line (no `module load` necessary):
```
[me@login-1.SAGA ~]$ singularity --version
singularity version 3.6.4-1.el7
```

The following examples will demonstrate how you can _run_ container images that has
already been prepared by others. If you want to learn how to _build_ your own containers,
see our code development {ref}`guides <dev-guides>`.

## Fetching images from the web

Singularity images can be fetched from the web using the `singularity pull` command,
which will download a SIF (Singularity Image Format) file to your current directory.
Notice that with Singularity, an image is just a simple binary file, and there's nothing
special about the directory in which you run the `singularity pull` command. This means
that you can move your image around as you please, and even `scp` it to a different
machine and execute it there (as long as you have Singularity installed, of course).

There are a number of different online repositories for hosting images, some of the
more common ones are listed below. Notice how you can pull Docker images
directly from Docker-Hub using Singularity.

Fetching from [Singularity-Hub](https://singularity-hub.org/):
```
$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```
Fetching from the [Sylabs](https://cloud.sylabs.io/library) library:
```
$ singularity pull --name alpine.sif library://alpine:latest
```
Fetching from [Docker-Hub](https://hub.docker.com/):
```
$ singularity pull --name alpine.sif docker://alpine:latest
```
Fetching from [Quay](https://quay.io/):
```
$ singularity pull --name openmpi-i8.sif docker://quay.io/bast/openmpi-i8:4.0.4-gcc-9.3.0
```

```{note}
The `singularity pull` step must be done on the login node. We cannot do this step
inside a job script because compute nodes cannot access the web.
```

## Hello world example

This example demonstrates:
1. how to download an image from an online repository, e.g. Singularity-Hub
2. how to run a container in a job script
3. how to execute a command from inside a container: `singularity exec <image-name>.sif <command>`
4. that the container runs it's own operating system (using the same kernel as the host)
5. that the current directory is mounted in the container by default, which means that it
will have access to input files etc. located in this directory (you will have read/write
permissions according to your own user)

First we pull a "hello world" Singularity image from Singularity-Hub:
```
$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```

This should be done on the login node, as noted above.
Once we have the SIF file, we can test it out with the following
job script on Saga (adjust `<myaccount>`; on Fram/Betzy you will need to remove
the line containing `#SBATCH --mem-per-cpu=1000M` but the rest should work as
is):

```bash
#!/bin/bash

#SBATCH --account=<myaccount>
#SBATCH --job-name=singularity-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=00:03:00

echo
echo "check that we can read the current directory from the container:"
singularity exec hello-world.sif ls

echo
echo "what is the operating system on the host?"
cat /etc/os-release

echo
echo "what is the operating system in the container?"
singularity exec hello-world.sif cat /etc/os-release
```

This produces the following output. Notice how in the container we are on a
Ubuntu operating system while the host is CentOS:
```
check that we can read the current directory from the container:
hello-world.sif
run.sh
slurm-1119935.out

what is the operating system on the host?
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

what is the operating system in the container?
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

