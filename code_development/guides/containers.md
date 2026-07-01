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
## What is a container image
In computing. a container image is a lightweight, standalone package that contains 
everything an application needs to run. Instead of installing software directly onto 
a computer and risking version conflicts with other apps, an image bundles the 
application code, runtime, system tools, libraries and configurations into a single, 
unchangeable file.

The defining character of a container is that it does not bundle a full operating system. 
Instead, it virtulizes at the software layer, sharing the host machine´s Linux kernel while
maintaing its own completely isolated user space.

## Understanding Container Image vs. Running Container
To use containers effectively, it is important to understand the distinction between an Image
and a Container.
- `The Container Image(The Blueprint)`:
This is a frozen, read-only file(such as a `.sif` file on our cluster). It acts as a blueprint
and contains all the static software layers, libraries and code, but it doesnot do anything on
its own.
- `The Running Container(The Instance)`:
This is the active process created when you execute an image(e.g. via `singularity shell`). It 
uses active system CPU and RAM to run your workloads. Running containers are transient, when 
your training job finishes or you exit the session, the container disappears. leaving the original
image completely unaltered. 

If you want a different environment, you don´t modify a running container but you simply use a
differnt image.

Hence, if you wrap your software stack inside a container, it solves the critical challenge of 
reproducibility. Every software dependecny from the base operating system and CUDA drivers,frameworks
and specific codes is locked down in a clear unchangeable record. Moreover, Containers abstract away
the underlying host hardware which allows build and test workflow on Windows, MAC  or Linux-based
HPC cluster.

Note: If you see a container image(`.sif`) in a shared directory inside the cluster, it is possible to start multiple container instances from it.Hence, many user can use the exact same file simultaneously. However, each user gets their own completely separate, active container instance running on different cluster nodes. None of them will interfere with each other and the original files remains completely untouched.

## Scenario where container is useful
Consider, you might have an older image classification project that relies on `NumPy v1.20`, 
but you want to start a new LLM fine-tuning project that requires `NumPy v2.0.` in your 
machine. Since, a standard operating system typically only allows one version of a library 
to be installed globally, the conflict rises.
1. Keeping the old version causes the new project to crash.
2. Upgrading the library fixes the new project but completely breaks the old code.

Hence, to control these version of these libraries within the runtime environment container
plays a key role. There is also another technology called Virtual Machines(VMs). While both 
Containers and Virtual Machines are virtualization technologies used to isolate environments,
they do so at completely different levels.

A Virtual Machine virtualizes a computer down to the hardware level. It allocates a chunk of 
physical CPU, RAM and storage and we need to install a full copy of Guest Operating System 
inside it. Because, it behaves like an entirely separate computer, it takes minutes to boot up
and consume massive system overhead just to idle. However, a Container skips the hardware 
emulation entirely. It hooks directly into the host machine´s existing Linux kernel for 
computing power, using a built-in Linux features (like namespaces and cgroups) to get its own
private workspace partition. Since, they only carry the specific libraries and application 
file they need, they launch instantly and run with near-zero performance loss which is ideal 
for HPC cluster.







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

## What is Singularity(Apptainer)?
Singularity(now developed under the open-source name Apptainer) is one of the container platform which is built specifically for HPC environments. The goal of Singularity is to package software and its dependencies into a single, portable unit. In the HPC infrastructure, security and performance are critical which is the reason why Singularity is chosen over another Docker.

It is very important to know why Docker is not used in HPC environment.Docker relies on a background service called the Docker Daemon, which runs with absoulute administrative privileges(`root`). This Docker Daemon will run the commands for the user when the regular user interact with Docker.Because of this, a user can easily exploit Docker to access, modify or delete system files on the host server which will be critical in terms of security.

However, Singularity completely eliminates this security risk by discarding the background daemon entirely.It operates entirely within the User Space with these functionalities:
1. The identity is strictly preserved. With the singularity container, you are exactly the same user inside and outside the container with the same permission that you have.
2. It launches just like any standard application or script on the cluster,
hooking directly into the host machine´s Linux Kernel using native isolation features(like namespaces) without needing administrative intervention. Hence , there are no background daemons like in Docker.
3. Singularity packages an entire runtime environment into a single Singularity Image Format(.sif) which is easy to copy and share.


## How to access singularity on NRIS HPC systems
Singularity is already installed globally on all our systems, and should be
immediately available on your command line (no `module load` necessary):
```console
[SAGA]$ singularity --version
singularity version 3.6.4-1.el7
```
## How to find container images
 - [Docker hub](https://hub.docker.com/)
 - [NVidia](https://ngc.nvidia.com/catalog/containers)
 - [Singularity Cloud](https://cloud.sylabs.io/library)
 - [Singularity Hub](https://singularity-hub.org/)
 - [RedHat](https://quay.io/)
 - [BioContainers](https://biocontainers.pro/)
 - [AMD](https://www.amd.com/en/developer/resources/infinity-hub.html)
 - From software developers


## How to get container images

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
We can inspect the image´s run script using the singularity inspect command `singularity inspect -r hello-world.sif` or simply as shown below.

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

With run, the default command is what is excuted that is embedded within the image.

[SAGA]$ singularity run hello-world.sif cat /etc/os-release
RaawwWWWWWRRRR!! Avocado!
So we need to use exec to get the expected result
[SAGA]$ singularity exec hello-world.sif cat /etc/os-release
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
...
..

```

## Building Singularity images
Building a Singularity container creates an unchaning, single-file snapshot of your 
software environment that can be tracked via version control alongside your research.
This ensures you can instantly rebuild or share the exact same runtime setup on any
machine, guaranteeing that your secientific results remain completely reproducible
years down the line. 

There are many ways to build a Singularity images. You can read more about them here 
[Build a container](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html).
However, we will discuss about one of the most commonly used approach which is building from a  
[Singularity Definition file](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html#building-containers-from-singularity-definition-files)

### Singularity Definition File
A singularity definition file is a text file that specifies the instructions and configuration
needed to build a container image.
Below is the simple illustration of definition file which only use the %post and %runscript sections.
More details can be found here [Definition File Offical Documentation](https://docs.sylabs.io/guides/3.7/user-guide/definition_files.html)
```bash
Bootstrap: docker
From: ubuntu:20.04

%post
  apt-get -y update && apt-get install -y python3

%runscript
  python3 -c 'print("Hello World!")'
```
The `%` prefix is used to define different configurations during the build process. In the above example,
we specify the backend protocolto fetch the base image as docker. This tells Singularity to pull the layers 
from a Docker registry ,automatically handles the download and convert it into Singularity´s format on the fly.
You cannot build a container using a competely blank file. Any software or code you write requires basic 
system utilities, core C libraries, and folder structures to interact with the system kernel. The most efficient
approach is to start with an existing Base Image that already provides this minimal operating system user space.
This is the reason why we pull a clean, minimal deployment of `Ubuntu 20.04`.

The `%post` block is where you actually install your applications, compile code and configure your container´s
environment.The command written in the `%post` section is an automated installation script. Every command here
is executed as standard shell code inside the base operating system you defined in the header. So, in this case 
here this command will run within your minimal Ubuntu 20.04 layout to update the package manager index and install
Python 3. We must bypass any interactive confirmation prompts `Do you want to continue? [Y/n]` by explicitly
forcing a "yes" response which is the reason why we passed `-y` flags. Beyond basic package managers, you can use
the `%post` section to run `pip install` commands, download remote datasets via `wget` or `curl`, create directories
or clone repositories. Everything modified here is permanently baked into the final, read-only `.sif` image. 

Finally, we have the `%runscript` section which is used to define a script that we want to run using the 
`singularity run ` command when a container is started. In this particular case, we just print the Hello World
message.

Once you save this `test.def` file you can build a `.sif` image using this command.
`apptainer build my_container.sif test.def`
```{note}
You cannot build it on the login node, so you have to build it on the compute node. Also, depending on the
implementation on each of our cluster, you might need to pass `--fakeroot` flag in the build command above if it 
fails.

```
## Interacting with Containers: Exec, Shell, and Host Integration
Once you have a container image (.sif), Singularity provides two primary ways to run commands inside it. You can either execute a single non-interactive command or drop into an interactive shell. We already saw above how we can use `exec` command to bypass the image´s default launch behaviour and executes the exact binary path.Moreover, it is also possible to open an interactive terminal inside the container to explore a container or debug an environment manually by using `shell` command. e.g. `singularity shell hello-world.sif`.

Singularity provides seamless host integration. If you run `whoami` or `ls` inside a newly pulled public container, you will notice that, it can already see your local files and also it knows your username. Singularity achieves this by:
1. Dynamic User Mapping:

When the container boots, it automatically reads your current user details from the host machines´s `/etc/passwd` and `/etc/group` files and injects them directly into the container. You remain exactly who you are, maintaining your identical system permissions.

2. Automatic Bind Mounts:

Singluarity automatically maps(binds) key directories from the host cluster directly into the container´s file structure. By default, your home directory `$HOME`, your current directory `$PWD`, and `/tmp` are made visible inside the container.Since this is not the copy, any files modified or saved while inside the container are written diretly to the host´s physical storage and will persist after the container shuts down.
However, if you want to access shared parallel cluster filesystems like `/scratch` or `/cluster/projects` you need to pass an explicit flag at runtime to make them visbible. `singularity shell --bind /scratch my-container.sif`. Refer to the section below to see how it works.

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



## Use Cases
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
see our code development {ref}`guides <dev-guides_containers>`.
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
job script on Saga (adjust `<myaccount>`; on Betzy you will need to remove
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



## Real world container examples

```{eval-rst}
.. toctree::
    :maxdepth: 1

    containers/bigdft.md
```
## Other notes
### singularity cache
Singularity handles images as standard `.sif` files on your disk. To save the network bandwidth and the time, it automatically maintains a local image cache.
When you pull an image from a remote registry (like Docker Hub or NVIDIA NGC), Singularity saves a copy in a hidden directory.If you accidently delete your local `.sif` file and try to pull it again, Singularity will instantly grab it from your local cache instead of downloading gigabytes of data over a network.

To inspect how much total space you are consuming, run this command `singularity cache list`. By default, this only shows the total file count and size. To see the specific images names, creation dates and where they were pulled from, add the verbose `-v` flag. `singularity cache list -v`

```{note}
Under the `TYPE` column, images from Singularity Hub appears as `shub`, while standard Docker/NGC images appear as `blob` or `library/oci`
```

Similarly, if your cluster storage quota is running low, you can clear out cached files using the `singularity cache clean`. Running this command will ask to wipe everything. However, if you want to see what exactly would be reomved without actually deleting it, you can use `-n` or `--dry-run` flag. Moreover, to wipe out only a specific type of cached image (e.g. clearing out old Singularity Hub templates while keeping your heavy Docker/NGC layers), you can use `singularity cache clean --type=shub` command
