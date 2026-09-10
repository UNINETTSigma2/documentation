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

- {ref}`What is Apptainer <what-is-apptainersingularity>`
- {ref}`How to access apptainer on NRIS HPC systems <how-to-access-apptainer-on-nris-hpc-systems>`
- {ref}`Getting container images <getting-container-images>`
- {ref}`Building Apptainer images <building-apptainer-images>`
- {ref}`Running a container <running-a-container>`
- {ref}`Additional Information <additional-information>`
- {ref}`Use cases <use-cases>`
- {ref}`Job scripts & examples <job-scripts-and-examples>`



(what-is-apptainersingularity)=
## What is Apptainer(Singularity)?
Apptainer (formerly Singularity) is one of the container platforms 
which is built specifically for HPC environments. The goal of 
Apptainer is to package software and its dependencies into a single, 
portable unit. In the HPC infrastructure, security and performance 
are critical which is the reason why Apptainer is chosen over another 
tool like Docker.

It is very important to know why Docker is not used in HPC environment. 
Docker relies on a background service called the Docker Daemon, which 
runs with absolute administrative privileges (`root`). This Docker 
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
Hence, there are no background daemons like in Docker.
3. Apptainer packages an entire runtime environment into a single Apptainer 
Image Format (.sif) which is easy to copy and share.


Please refer to the {ref}`Additional Information <additional-information>` to learn more about containers.


(how-to-access-apptainer-on-nris-hpc-systems)=
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

(getting-container-images)=
## Getting container images

Before getting a container images to run inside the cluster, you need 
to know where to find these container images that you can use for your 
project. Here is the list of sources where you could find a container image from:
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
For both Saga and Betzy, the apptainer pull command will work without specifying the architecture. However, on Olivia the architecture is different in the login nodes and the compute nodes. Hence, you must use this `--arch arm64` flag which specify the architecture in the apptainer pull command.
```

```{note}
To prevent home-directory quota issues, please do these before pulling the larger image so that, the cache will be stored in your project area.

1. `mkdir -p /cluster/work/projects/<project_number>/$USER/apptainer/tmp`
2. `export APPTAINER_TMPDIR=/cluster/work/projects/<project_number>/apptainer/tmp`
3. `export APPTAINER_CACHEDIR=/cluster/work/projects/<project_number>/$USER/apptainer/tmp`

```

(building-apptainer-images)=
## Building Apptainer images

In many cases, you might want to build your own container images and then install 
additional packages into the image. Please refer to this {ref}`Building Container <building-container>` 
section to read more about it.

(running-a-container)=
## Running a container
Once you have a container image (.sif), Apptainer provides three primary ways to interact with it depending on your needs:

1. `run`: Executes the container's default action or entry point script (defined when the image was built).

2. `exec`: Bypasses the image’s default launch behavior to execute a specific binary or custom command.

3. `shell`: Opens an interactive terminal inside the container for manual exploration and debugging.

If you pulled the `hello-world.sif` image earlier, you can use the `exec` command to run a specific command inside it like this:

```bash
apptainer exec hello-world.sif echo "Hello from inside the container!"
```


Moreover, it is also possible to open an interactive terminal inside the container to explore the environment or debug manually by using the shell command:

```bash
apptainer shell hello-world.sif
``` 

To seamlessly work with your data, Apptainer automatically links specific folders from your host computer into the container at startup. This feature, known as automatic bind mounting, ensures you can read and write files without losing access to your personal storage.

If you want to read more about how Apptainer provides host integration, the deep differences between run, exec, and shell, exactly which system paths are automatically bound, and how you can manually map custom folders to access your host files, refer to {ref}`apptainer-info` section.



(use-cases)=   
## Use Cases
Please refer to this page to read more about the {ref}`Use Cases <use-cases-nris>` where 
the containers might be useful in the NRIS system.

(job-scripts-and-examples)=
## Job script and Examples
If you want to read more about how we can use the apptainer in the job scripts, 
please refer to this {ref}`Job Script Apptainer <job-script-apptainer>`






```{toctree}
:hidden:

containers/building-container
containers/job-scripts-examples
containers/advanced-use-cases
containers/bigdft
containers/additional-information/understanding-container-concepts
containers/additional-information/container-benefit
containers/additional-information/singularity-cache
containers/additional-information/apptainer-info
containers/additional-information/additional-info
```