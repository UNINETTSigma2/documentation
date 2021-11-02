(running-containers)=
# Containers on NRIS HPC systems

```{note}
Currently, [Singularity](https://sylabs.io/singularity/) is the only supported container
solution on our HPC systems (Saga, Fram, Betzy). However, since Singularity can build
containers from Docker images, it is also possible to run [Docker](https://www.docker.com/)
containers through Singularity.
```
## What is a container image
Container image is a package with all cargo needed for a software to work. This
includes the operating system, system packages, libraries and applications as
a single unit. It only uses the host operating systems kernel.     


## What is not covered in this document
We are showing how to use existing container images on our systems as regular users.
Operations that require root access, like building or making changes to an images
not discussed. 

## When to use containers on NRIS HPC systems

```{note}
Please let us know if you find more reasons for using containers
```
 - If you have a software stack or a pipeline already setup somewhere else and 
   you want to bring it as it is to one of the HPC systems
     - Containers give the users the flexibility to bring a full software stack to the
       cluster which has already been set up, which can make software installations and
       dependencies more reproducible and more portable across clusters.
 - The software you want to use is only available as a container image
 - You need to use system level installations, e.g. the procedure involved
   `apt-get install SOMETHING` or similar (`yum`, `rpm`, etc).
 - You have a old software that needs some older dependencies.
 - You need a specific version of a software to run another software, e.g. CUDA.

## When not to use containers on NRIS HPC systems
 - If the software you are planing to to use already installed as a module(s), then
   better to use that module or collection of modules
 - Windows containers. On NRIS HPC systems only containers that uses UNIX kernel would 
   work 
 - If you do not know what the container is exactly for. i.e. found a command on
   the internet and just want to try it out

## How to access singularity on NRIS HPC systems
Singularity is already installed globally on all our systems, and should be
immediately available on your command line (no `module load` necessary):
```console
[SAGA]$ singularity --version
singularity version 3.6.4-1.el7
```



## How to find container images

Singularity images can be fetched from the web using the `singularity pull` command,
which will download a SIF (Singularity Image Format) file to your current directory.
Notice that with Singularity, an image is just a simple binary file, and there's nothing
special about the directory in which you run the `singularity pull` command. This means
that you can move your image around as you please, and even `scp` it to a different
machine and execute it there (as long as you have Singularity installed, of course).

There are a number of different online repositories for hosting images, some of the
more common ones are listed below. Notice how you can pull Docker images
directly from Docker-Hub using Singularity.

```console
#Fetching from a [Singularity](https://singularityhub.github.io/) registry:
$ singularity pull --name hello-world.sif shub://vsoch/hello-world

#Fetching from a [Sylabs](https://cloud.sylabs.io/library) registry:
$ singularity pull --name alpine.sif library://alpine:latest

#Fetching from a [Docker-Hub](https://hub.docker.com/) registry:
$ singularity pull --name alpine.sif docker://alpine:latest

#Fetching from a [Quay](https://quay.io) registry:
$ singularity pull --name openmpi-i8.sif docker://quay.io/bast/openmpi-i8:4.0.4-gcc-9.3.0
```

```{note}

 singularity run Vs singularity exec
- [singularity exec](https://sylabs.io/guides/3.1/user-guide/cli/singularity_exec.html):
   Run a command within a container
- [singularity run](https://sylabs.io/guides/3.1/user-guide/cli/singularity_run.html):
   Run the user-defined default command within a container	

```

Example 
```console

[SAGA]$ singularity pull --name hello-world.sif shub://vsoch/hello-world
The image created (hello-world.sif) has a user defined command called "rawr.sh"

[SAGA]$ singularity exec hello-world.sif cat /singularity
#!/bin/sh 

exec /bin/bash /rawr.sh
[SAGA]$ singularity exec hello-world.sif cat /rawr.sh
#!/bin/bash

echo "RaawwWWWWWRRRR!! Avocado!"


[SAGA]$ singularity run hello-world.sif 
RaawwWWWWWRRRR!! Avocado!

With run, the default command is what is excuted,
[SAGA]$ singularity run hello-world.sif cat /etc/os-release
RaawwWWWWWRRRR!! Avocado!
So we need to use exec to get the expected result
[SAGA]$ singularity exec hello-world.sif cat /etc/os-release
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
...
..

```


Following are some example use cases we have seen on NRIS HPC systems. 

```{note}
Example 1: A user wants to use different version of the TensorFlow  than 
what is installed in SAGA. So she googles and ends up here
[https://www.tensorflow.org/install](https://www.tensorflow.org/install)
There she finds the following command sequence
```
```console
 docker pull tensorflow/tensorflow:latest  # Download latest stable image
 docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server 
```
But she knows that we do not have Docker on SAGA so she uses Singularity to pull 
the image, yes it is possible to pull docker images using singularity
```console
[SAGA]$ singularity pull docker://tensorflow/tensorflow:latest
```
To test she prints the version
```console
[SAGA]$ singularity run  tensorflow_latest.sif python -c "import tensorflow as tf;print(tf.__version__)" 
```


```{note}
Example 2:
A user needs to use a software that runs only on a specific vesion of Ubuntu
```

```console
[SAGA]$ singularity pull docker://bioperl/bioperl
[SAGA]$ singularity exec  bioperl_latest.sif cat /etc/os-release                    
    NAME="Ubuntu"
    VERSION="14.04.5 LTS, Trusty Tahr"

[SAGA]$ singularity exec  bioperl_latest.sif perl -e 'use Bio::SeqIO; print join "\n", %INC; print "\n"'
    base.pm
    /usr/share/perl/5.18/base.pm
    File/Path.pm
    /usr/share/perl/5.18/File/Path.pm


```
```{warning}
If a ready made image is not available with the software. Then you need to pull
Ubuntu image to a machine where you have root access, install the software,
repackage it and take it to SAGA. This step is not covered here.
If you want to learn how to _build_ your own containers,
see our code development {ref}`guides <dev-guides>`.
```

## Singularity in Job scripts

This example demonstrates:
1. how to run a container in a job script
2. how to execute a command from inside a container: `singularity exec <image-name>.sif <command>`
3. that the container runs it's own operating system (using the same kernel as the host)
4. that your `$HOME` directory is mounted in the container by default, which means that it
will have access to input files etc. located somewhere in this directory (you will have read/write
permissions according to your own user)

First we pull a "hello world" Singularity image from Singularity-Hub. This we need
to do from the login node, before the job is submitted. i.e. we do not pull
images from within a job.
```console
[SAGA]$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```

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

```{note}
The behavior described in the above example is only accurate if you run it from somewhere within your `$HOME`
directory. If you run it from somewhere else, like `/cluster/projects/` or `/cluster/work/` you will *not*
enter the container environment from the current directory, but rather from your root `$HOME` directory, i.e.
the output from the first `ls` command in the script will be equivalent to `$ ls $HOME`. If you want to access
files that are *not* located in your `$HOME` you'll need to `--bind` that directory explicitly as described below.
```

## Access project area from the container

Singularity container can access the home directory
but to access the project directory we need to bind it first.

Lets try it out
```console
[SAGA]$ head -n2 data/input.txt
1
2
[SAGA]$ singularity exec hello-world.sif  head -n2 data/input.txt
/usr/bin/head: cannot open 'data/input.txt' for reading: No such file or directory
[SAGA]$ pwd
/cluster/projects/nnxxxxk/containers
[SAGA]$ singularity exec hello-world.sif  head -n2 /cluster/projects/nnxxxxk/containers/data/input.txt
/usr/bin/head: cannot open '/cluster/projects/nnxxxxk/containers/data/input.txt' for reading: No such file or directory
```

Now we use binding to attach local storage and then the container would have access.

```console
[SAGA]$ singularity exec --bind /path/containers/data:/data bioperl_latest.sif head -n2 /data/input.txt
1
2

```

## Real world container examples

- [BigDFT with MPI and CUDA](containers/bigdft.md)

## Other notes
### singularity cache
