---
orphan: true
---

(running-containers)=

# Containers on NRIS HPC systems
```{note}
Currently, [Apptainer](https://apptainer.org/) is the only supported 
container solution on our HPC systems (Saga, Betzy, Olivia). However, 
since Apptainer can build containers from Docker images, it is also 
possible to run [Docker](https://www.docker.com/) containers through 
Apptainer. Moreover, for this particular guide, we assume the reader 
should have some prior knowledge about docker.
```
## Table of Contents

- [What is Apptainer](#what-is-apptainersingularity)
- [How to access apptainer on NRIS HPC systems](#how-to-access-apptainer-on-nris-hpc-systems)
- [Getting container images](#getting-container-images)
- [Building Apptainer images](#building-apptainer-images)
- [Running a container](#running-a-container)
- {ref}`additional-information`
- [Use cases](#use-cases)
- [Job scripts & examples](#job-scripts-and-examples)




## What is Apptainer(Singularity)?
Apptainer (formally Singularity) is one of the container platform 
which is built specifically for HPC environments. The goal of 
Apptainer is to package software and its dependencies into a single, 
portable unit. In the HPC infrastructure, security and performance 
are critical which is the reason why Apptainer is chosen over another 
tool like Docker.

It is very important to know why Docker is not used in HPC environment. 
Docker relies on a background service called the Docker Daemon, which 
runs with absoulute administrative privileges (`root`). This Docker 
Daemon will run the commands for the user when the regular user interact 
with Docker. Because of this, a user can easily exploit Docker to access, 
modify or delete system files on the host server which will be critical 
in terms of security.

However, Apptainer completely eliminates this security risk by discarding 
the background daemon entirely. It operates entirely within the User Space 
with these functionalities:
1. The identity is strictly preserved. With the apptainer container, you 
are exactly the same user inside and outside the container with the same 
permission that you have.
2. It launches just like any standard application or script on the cluster,
hooking directly into the host machine´s Linux Kernel using native isolation 
features (like namespaces) without needing administrative intervention. 
Hence , there are no background daemons like in Docker.
3. Apptainer packages an entire runtime environment into a single Apptainer 
Image Format(.sif) which is easy to copy and share.


Please refer to the {ref}`additional-information` to learn more about containers.

### How to access apptainer on NRIS HPC systems
Apptainer is already installed globally on all our systems, and should be
immediately available on your command line (no `module load` necessary):

```{eval-rst}
.. tabs::

   .. group-tab:: Saga

      .. code-block:: console

         [SAGA]$ apptainer --version
         apptainer version 1.4.4-1.el9

   .. group-tab:: Betzy

      .. code-block:: console

         [BETZY]$ apptainer --version
         apptainer version 1.4.4-1.el9

   .. group-tab:: Olivia

      .. code-block:: console

         [OLIVIA]$ apptainer --version
         apptainer version 1.4.5-150600.4.12.1
```

## Getting container images

Before getting a container images to run inside the cluster, you need 
to know where to find these container images that you can use for your 
project. Here are list of sources where you could find a container image from:
1. [Docker](https://hub.docker.com/)
2. [Nvidia NGC catalog](https://catalog.ngc.nvidia.com/search?sort=weightPopularDESC&resourceType=container)
3. [Singularity Hub](https://datasets.datalad.org/?dir=/shub)

Once you know where to fetch it from, Apptainer images can be fetched 
from the web using the `apptainer pull` command, which will download a 
SIF (Singularity Image Format) file to your current directory.
Notice that with Apptainer, an image is just a simple binary file, and 
there's nothing special about the directory in which you run the `apptainer pull` 
command. This means that you can move your image around as you please, 
and even `scp` it to a different machine and execute it there (as long as 
you have Apptainer installed, of course).

There are a number of different online repositories for hosting images, 
some of the more common ones are listed below. Notice how you can pull 
Docker images directly from Docker-Hub using Apptainer.

```{eval-rst}
.. tabs::

   .. group-tab:: Saga

      .. code-block:: console
         
         apptainer pull --name hello-world.sif shub://vsoch/hello-world

   .. group-tab:: Betzy

      .. code-block:: console

         apptainer pull --name ubuntu.sif docker://quay.io/libpod/ubuntu:latest

   .. group-tab:: Olivia

      .. code-block:: console

         apptainer pull --arch arm64 pytorch_25.08_cuda13.0_arm.sif docker://nvcr.io/nvidia/pytorch:25.08-py3
```

```{note}
To prevent home-directory quota issues, please do these before pulling the larger image so that, the cache will be stored in your project area.

1. mkdir -p /cluster/work/projects/<project_number>/$USER/apptainer/tmp
2. export APPTAINER_TMPDIR=/cluster/work/projects/<project_number>/apptainer/tmp
3. export APPTAINER_CACHEDIR=/cluster/work/projects/<project_number>/$USER/apptainer/tmp

```

## Building Apptainer images

In many cases, you might want to build your own container images and then install 
additional packages into the image. Please refer to this {ref}`building-container` 
section to read more about it.

## Running a container

Once you have a container image (.sif), Apptainer provides two primary ways to run 
commands inside it. You can either execute a single non-interactive command or drop 
into an interactive shell. You can use `exec` command to bypass the image´s default 
launch behaviour and executes the exact binary path.So, if you have `hello-world.sif` 
image that you pull earlier, then you can use, 
```bash
apptainer exec helloworld.sif
```  
command to run it. Moreover, it is also possible to open an interactive terminal 
inside the container to explore a container or debug an environment manually by 
using `shell` command. e.g. 
```bash
apptainer shell hello-world.sif
```

If you want to read more about how apptainer provides host integration, difference 
between `exec` and `shell` , what are automatically binded and how you can access 
your files from the host, refer to {ref}`apptainer-info` section.

Moreover, please refer to the example in the same page to see how binding 
actually work in practice.



## Use Cases
Please refer to this page to read more about the {ref}`use-cases-nris` where 
the containers might be useful in the NRIS system.


## Job scripts and Examples
If you want to read more about how we can use the apptainer in the job scripts, 
please refer to this {ref}`job-script-apptainer`


