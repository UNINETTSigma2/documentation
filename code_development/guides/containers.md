---
orphan: true
---

(running-containers)=

# Containers on NRIS HPC systems

```{note}
Currently, [Singularity](https://sylabs.io/singularity/) is the only supported container
solution on our HPC systems (Saga, Betzy, Olivia). However, since Singularity can build
containers from Docker images, it is also possible to run [Docker](https://www.docker.com/)
containers through Singularity.

```

```{contents} __Page Overview__
:depth: 2
```

## What is a container image

In computing, a container image is a lightweight, standalone package that contains everything an application needs to run. Instead of installing software directly onto a computer and risking version conflicts with other apps, an image bundles the application code, runtime, system tools, libraries, and configurations into a single, unchangeable file.

The defining characteristic of a container is that it does not bundle a full operating system. Instead, it virtualizes at the software layer, sharing the host machine's Linux kernel while maintaining its own completely isolated user space.

## Understanding Container Image vs. Running Container

To use containers effectively, it is important to understand the distinction between an Image and a Container.

* `The Container Image (The Blueprint)`: This is a frozen, read-only file (such as a `.sif` file on our cluster). It acts as a blueprint and contains all the static software layers, libraries, and code, but it does not do anything on its own.
* `The Running Container (The Instance)`: This is the active process created when you execute an image (e.g., via `singularity shell`). It uses active system CPU and RAM to run your workloads. Running containers are transient; when your training job finishes or you exit the session, the container disappears, leaving the original image completely unaltered.

If you want a different environment, you don't modify a running container; you simply use a different image.

Hence, wrapping your software stack inside a container solves the critical challenge of reproducibility. Every software dependency, from the base operating system and CUDA drivers to frameworks and specific codes, is locked down in a clear, unchangeable record. Moreover, containers abstract away the underlying host hardware, which allows you to build and test workflows on a Windows, Mac, or Linux-based HPC cluster.

Note: If you see a container image (`.sif`) in a shared directory inside the cluster, it is possible to start multiple container instances from it. Hence, many users can use the exact same file simultaneously. However, each user gets their own completely separate, active container instance running on different cluster nodes. None of them will interfere with each other, and the original file remains completely untouched.

## Scenarios where containers are useful

Consider this scenario: you might have an older image classification project that relies on `NumPy v1.20`, but you want to start a new LLM fine-tuning project that requires `NumPy v2.0` on your machine. Since a standard operating system typically only allows one version of a library to be installed globally, a conflict arises.

1. Keeping the old version causes the new project to crash.
2. Upgrading the library fixes the new project but completely breaks the old code.

Hence, containers play a key role in controlling the version of these libraries within the runtime environment. There is also another technology called Virtual Machines (VMs). While both Containers and Virtual Machines are virtualization technologies used to isolate environments, they do so at completely different levels.

A Virtual Machine virtualizes a computer down to the hardware level. It allocates a chunk of physical CPU, RAM, and storage, and requires installing a full copy of a Guest Operating System inside it. Because it behaves like an entirely separate computer, it takes minutes to boot up and consumes massive system overhead just to idle. However, a Container skips hardware emulation entirely. It hooks directly into the host machine's existing Linux kernel for computing power, using built-in Linux features (like namespaces and cgroups) to get its own private workspace partition. Since they only carry the specific libraries and application files they need, they launch instantly and run with near-zero performance loss, which is ideal for an HPC cluster.

## When to use containers on NRIS HPC systems

```{note}
Please let us know if you find more reasons for using containers.

```

* If you have a software stack or a pipeline already set up somewhere else and you want to bring it as it is to one of the HPC systems.
* Containers give users the flexibility to bring a full software stack to the cluster that has already been set up, which can make software installations and dependencies more reproducible and more portable across clusters.


* The software you want to use is only available as a container image.
* You need to use system-level installations, e.g., the procedure involves `apt-get install SOMETHING` or similar (`yum`, `rpm`, etc.).
* You have old software that needs older dependencies.
* You need a specific version of a software dependency to run another software, e.g., CUDA.

## When not to use containers on NRIS HPC systems

* If the software you are planning to use is already installed as a module(s), then it is better to use that module or collection of modules.
* Windows containers. On NRIS HPC systems, only containers that use a UNIX kernel will work.
* If you do not know exactly what the container is for—i.e., you found a command on the internet and just want to try it out.

## What is Singularity (Apptainer)?

Singularity (now developed under the open-source name Apptainer) is a container platform built specifically for HPC environments. The goal of Singularity is to package software and its dependencies into a single, portable unit. In an HPC infrastructure, security and performance are critical, which is why Singularity is chosen over Docker.

It is very important to know why Docker is not used in HPC environments. Docker relies on a background service called the Docker Daemon, which runs with absolute administrative privileges (`root`). This Docker Daemon runs commands for the user when a regular user interacts with Docker. Because of this, a user can easily exploit Docker to access, modify, or delete system files on the host server, which is critical in terms of security.

However, Singularity completely eliminates this security risk by discarding the background daemon entirely. It operates entirely within User Space with these functionalities:

1. Identity is strictly preserved. With a Singularity container, you are exactly the same user inside and outside the container, with the same permissions you normally have.
2. It launches just like any standard application or script on the cluster, hooking directly into the host machine's Linux Kernel using native isolation features (like namespaces) without needing administrative intervention. Hence, there are no background daemons like in Docker.
3. Singularity packages an entire runtime environment into a single Singularity Image Format (`.sif`) file, which is easy to copy and share.

## How to access singularity on NRIS HPC systems

Singularity is already installed globally on all our systems and should be immediately available on your command line (no `module load` necessary):

```console
[SAGA]$ singularity --version
singularity version 3.6.4-1.el7

```

## How to find container images

* [Docker Hub](https://hub.docker.com/)
* [NVIDIA](https://ngc.nvidia.com/catalog/containers)
* [Singularity Cloud](https://cloud.sylabs.io/library)
* [Singularity Hub](https://singularity-hub.org/)
* [RedHat](https://quay.io/)
* [BioContainers](https://biocontainers.pro/)
* [AMD](https://www.amd.com/en/developer/resources/infinity-hub.html)
* From software developers

## How to get container images

Singularity images can be fetched from the web using the `singularity pull` command, which will download a SIF (Singularity Image Format) file to your current directory. Notice that with Singularity, an image is just a simple binary file, and there's nothing special about the directory in which you run the `singularity pull` command. This means that you can move your image around as you please, and even `scp` it to a different machine and execute it there (as long as you have Singularity installed, of course).

There are a number of different online repositories for hosting images; some of the more common ones are listed below. Notice how you can pull Docker images directly from Docker Hub using Singularity.

```console
# Fetching from a Singularity registry:
$ singularity pull --name hello-world.sif shub://vsoch/hello-world

# Fetching from a Sylabs registry:
$ singularity pull --name alpine.sif library://alpine:latest

# Fetching from a Docker Hub registry:
$ singularity pull --name alpine.sif docker://alpine:latest

# Fetching from a Quay registry:
$ singularity pull --name openmpi-i8.sif docker://quay.io/bast/openmpi-i8:4.0.4-gcc-9.3.0

```

```{note}
singularity run vs singularity exec
- [singularity exec](https://sylabs.io/guides/3.1/user-guide/cli/singularity_exec.html): Run a command within a container
- [singularity run](https://sylabs.io/guides/3.1/user-guide/cli/singularity_run.html): Run the user-defined default command within a container

```

We can inspect the image's run script using the singularity inspect command `singularity inspect -r hello-world.sif` or simply as shown below.

### Example

```console
[SAGA]$ singularity pull --name hello-world.sif shub://vsoch/hello-world
The image created (hello-world.sif) has a user-defined command called "rawr.sh"

[SAGA]$ singularity exec hello-world.sif cat /singularity
#!/bin/sh

exec /bin/bash /rawr.sh

[SAGA]$ singularity exec hello-world.sif cat /rawr.sh
#!/bin/bash

echo "RaawwWWWWWRRRR!! Avocado!"

[SAGA]$ singularity run hello-world.sif
RaawwWWWWWRRRR!! Avocado!

With run, the default command embedded within the image is what is executed.

[SAGA]$ singularity run hello-world.sif cat /etc/os-release
RaawwWWWWWRRRR!! Avocado!

So we need to use exec to get the expected result:
[SAGA]$ singularity exec hello-world.sif cat /etc/os-release
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
...
..

```

## Building Singularity images

Building a Singularity container creates an unchangeable, single-file snapshot of your software environment that can be tracked via version control alongside your research. This ensures you can instantly rebuild or share the exact same runtime setup on any machine, guaranteeing that your scientific results remain completely reproducible years down the line.

There are many ways to build Singularity images. You can read more about them here: [Build a container](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html). However, we will discuss one of the most commonly used approaches, which is building from a [Singularity Definition file](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html#building-containers-from-singularity-definition-files).

### Singularity Definition File

A Singularity definition file is a text file that specifies the instructions and configuration needed to build a container image. Below is a simple illustration of a definition file that only uses the `%post` and `%runscript` sections. More details can be found here: [Definition File Official Documentation](https://docs.sylabs.io/guides/3.7/user-guide/definition_files.html).

```bash
Bootstrap: docker
From: ubuntu:20.04

%post
  apt-get -y update && apt-get install -y python3

%runscript
  python3 -c 'print("Hello World!")'

```

The `%` prefix is used to define different configurations during the build process. In the above example, we specify the backend protocol to fetch the base image as `docker`. This tells Singularity to pull the layers from a Docker registry, automatically handle the download, and convert it into Singularity's format on the fly.

You cannot build a container using a completely blank file. Any software or code you write requires basic system utilities, core C libraries, and folder structures to interact with the system kernel. The most efficient approach is to start with an existing Base Image that already provides this minimal operating system user space. This is the reason why we pull a clean, minimal deployment of `Ubuntu 20.04`.

The `%post` block is where you actually install your applications, compile code, and configure your container's environment. The commands written in the `%post` section function as an automated installation script. Every command here is executed as standard shell code inside the base operating system you defined in the header. So, in this case, the commands will run within your minimal Ubuntu 20.04 layout to update the package manager index and install Python 3. We must bypass any interactive confirmation prompts like `Do you want to continue? [Y/n]` by explicitly forcing a "yes" response, which is why we pass the `-y` flag. Beyond basic package managers, you can use the `%post` section to run `pip install` commands, download remote datasets via `wget` or `curl`, create directories, or clone repositories. Everything modified here is permanently baked into the final, read-only `.sif` image.

Finally, we have the `%runscript` section, which is used to define a script that we want to run using the `singularity run` command when a container is started. In this particular case, we just print a "Hello World!" message.

Once you save this `test.def` file, you can build a `.sif` image using this command:
`apptainer build my_container.sif test.def`

```{note}
You cannot build it on the login node, so you have to build it on a compute node. Also, depending on the implementation on each cluster, you might need to pass the `--fakeroot` flag in the build command above if it fails.

```

## Interacting with Containers: Exec, Shell, and Host Integration

Once you have a container image (`.sif`), Singularity provides two primary ways to run commands inside it. You can either execute a single non-interactive command or drop into an interactive shell. We already saw above how we can use the `exec` command to bypass the image's default launch behavior and execute an exact binary path. Moreover, it is also possible to open an interactive terminal inside the container to explore it or debug an environment manually by using the `shell` command (e.g., `singularity shell hello-world.sif`).

Singularity provides seamless host integration. If you run `whoami` or `ls` inside a newly pulled public container, you will notice that it can already see your local files and also knows your username. Singularity achieves this by:

1. **Dynamic User Mapping**: When the container boots, it automatically reads your current user details from the host machine's `/etc/passwd` and `/etc/group` files and injects them directly into the container. You remain exactly who you are, maintaining identical system permissions.
2. **Automatic Bind Mounts**: Singularity automatically maps (binds) key directories from the host cluster directly into the container's file structure. By default, your home directory `$HOME`, your current directory `$PWD`, and `/tmp` are made visible inside the container. Since this is not a copy, any files modified or saved while inside the container are written directly to the host's physical storage and will persist after the container shuts down.

However, if you want to access shared parallel cluster filesystems like `/scratch` or `/cluster/projects`, you need to pass an explicit flag at runtime to make them visible: `singularity shell --bind /scratch my-container.sif`. Refer to the section below to see how it works.

## Accessing project areas from the container

A Singularity container can access your home directory by default, but to access a project directory, we need to bind it first.

Let's try it out:

```console
[SAGA]$ head -n2 data/input.txt
1
2
[SAGA]$ singularity exec hello-world.sif head -n2 data/input.txt
/usr/bin/head: cannot open 'data/input.txt' for reading: No such file or directory
[SAGA]$ pwd
/cluster/projects/nnxxxxk/containers
[SAGA]$ singularity exec hello-world.sif head -n2 /cluster/projects/nnxxxxk/containers/data/input.txt
/usr/bin/head: cannot open '/cluster/projects/nnxxxxk/containers/data/input.txt' for reading: No such file or directory

```

Now we use binding to attach local storage, giving the container access:

```console
[SAGA]$ singularity exec --bind /path/containers/data:/data bioperl_latest.sif head -n2 /data/input.txt
1
2

```

## Use Cases

Following are some example use cases we have seen on NRIS HPC systems.

```{note}
Example 1: A user wants to use a different version of TensorFlow than what is installed on SAGA. So she googles and ends up here: [https://www.tensorflow.org/install](https://www.tensorflow.org/install).
There she finds the following command sequence:

```

```console
docker pull tensorflow/tensorflow:latest  # Download latest stable image
docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server

```

But she knows that we do not have Docker on SAGA, so she uses Singularity to pull the image (yes, it is possible to pull Docker images using Singularity):

```console
[SAGA]$ singularity pull docker://tensorflow/tensorflow:latest

```

To test, she prints the version:

```console
[SAGA]$ singularity run tensorflow_latest.sif python -c "import tensorflow as tf;print(tf.__version__)"

```

```{note}
Example 2: A user needs to use software that runs only on a specific version of Ubuntu.

```

```console
[SAGA]$ singularity pull docker://bioperl/bioperl
[SAGA]$ singularity exec bioperl_latest.sif cat /etc/os-release
    NAME="Ubuntu"
    VERSION="14.04.5 LTS, Trusty Tahr"

[SAGA]$ singularity exec bioperl_latest.sif perl -e 'use Bio::SeqIO; print join "\n", %INC; print "\n"'
    base.pm
    /usr/share/perl/5.18/base.pm
    File/Path.pm
    /usr/share/perl/5.18/File/Path.pm

```

```{warning}
If a ready-made image is not available with the software you need, you will need to pull an Ubuntu image to a machine where you have root access, install the software, repackage it, and bring it to SAGA. This step is not covered here.
If you want to learn how to *build* your own containers, see our code development {ref}`guides <dev-guides_containers>`.

```

## Singularity in Job scripts

This example demonstrates:

1. How to run a container in a job script.
2. How to execute a command from inside a container: `singularity exec <image-name>.sif <command>`.
3. That the container runs its own operating system (while using the same kernel as the host).
4. That your `$HOME` directory is mounted in the container by default, which means that it will have access to input files, etc., located somewhere in this directory (you will have read/write permissions matching your own user account).

First, we pull a "hello world" Singularity image from Singularity Hub. This needs to be done from the login node before the job is submitted—i.e., we do not pull images from within a job script.

```console
[SAGA]$ singularity pull --name hello-world.sif shub://vsoch/hello-world

```

Once we have the SIF file, we can test it out with the following job script on Saga (adjust `<myaccount>`; on Betzy you will need to remove the line containing `#SBATCH --mem-per-cpu=1000M`, but the rest should work as is):

```bash
#!/bin/bash

#SBATCH --account=<myaccount>
#SBATCH --job-name=singularity-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=00:03:00


echo
echo "what is the operating system on the host?"
cat /etc/os-release

echo
echo "what is the operating system in the container?"
singularity exec hello-world.sif cat /etc/os-release

```

This produces the following output. Notice how inside the container we are on an Ubuntu operating system, while the host is CentOS:

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
The behavior described in the above example is only accurate if you run it from somewhere within your `$HOME` directory. If you run it from somewhere else, like `/cluster/projects/` or `/cluster/work/`, you will *not* enter the container environment from the current directory, but rather from your root `$HOME` directory—i.e., the output from the first `ls` command in the script will be equivalent to `$ ls $HOME`. If you want to access files that are *not* located in your `$HOME`, you'll need to `--bind` that directory explicitly as described below.

```

## Real world container examples

```{eval-rst}
.. toctree::
    :maxdepth: 1

    containers/bigdft.md

```

## Other notes

### singularity cache

Singularity handles images as standard `.sif` files on your disk. To save network bandwidth and time, it automatically maintains a local image cache. When you pull an image from a remote registry (like Docker Hub or NVIDIA NGC), Singularity saves a copy in a hidden directory. If you accidentally delete your local `.sif` file and try to pull it again, Singularity will instantly grab it from your local cache instead of downloading gigabytes of data over the network again.

To inspect how much total space you are consuming, run this command: `singularity cache list`. By default, this only shows the total file count and size. To see the specific image names, creation dates, and where they were pulled from, add the verbose `-v` flag: `singularity cache list -v`.

```{note}
Under the `TYPE` column, images from Singularity Hub appear as `shub`, while standard Docker/NGC images appear as `blob` or `library/oci`.

```

Similarly, if your cluster storage quota is running low, you can clear out cached files using the `singularity cache clean` command. Running this command will ask to wipe everything. However, if you want to see exactly what would be removed without actually deleting it, you can use the `-n` or `--dry-run` flag. Moreover, to wipe out only a specific type of cached image (e.g., clearing out old Singularity Hub templates while keeping your heavy Docker/NGC layers), you can use the `singularity cache clean --type=shub` command.
