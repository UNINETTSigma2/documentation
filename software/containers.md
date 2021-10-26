(running-containers)=
# Running containers

```{note}
Currently, [Singularity](https://sylabs.io/singularity/) is the only supported container
solution on our HPC systems (Saga, Fram, Betzy). However, since Singularity can build
containers from Docker images, it is also possible to run [Docker](https://www.docker.com/)
containers through Singularity.
```

# When to use containers on NRIS HPC systems

```{note}
Please let us know if you find more reasens for using containers
```
 - Containers give the users the flexibility to bring a full software stack to the
   cluster which has already been set up, which can make software installations and
   dependencies more reproducible and more portable across clusters.


# How to access singularity on NRIS HPC systems
Singularity is already installed globally on all our systems, and should be
immediately available on your command line (no `module load` necessary):
```
[me@login-1.SAGA ~]$ singularity --version
singularity version 3.6.4-1.el7
```
Please not that there are no modules to be loaded as singularity is installed
system wide (on login and compute nodes)

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

Fetching from a [Singularity](https://singularityhub.github.io/) registry:
```
$ singularity pull --name hello-world.sif shub://vsoch/hello-world
```
Fetching from a [Sylabs](https://cloud.sylabs.io/library) registry:
```
$ singularity pull --name alpine.sif library://alpine:latest
```
Fetching from a [Docker-Hub](https://hub.docker.com/) registry:
```
$ singularity pull --name alpine.sif docker://alpine:latest
```
Fetching from a [Quay](https://quay.io) registry:
```
$ singularity pull --name openmpi-i8.sif docker://quay.io/bast/openmpi-i8:4.0.4-gcc-9.3.0
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

(bigdft-cuda-example)=
## Real world example: BigDFT

```{note}
Parts of the following example require access to NVIDIA GPU resources. It has been tested
successfully on Saga, but there's no guarantee it will work seamlessly on other systems.
```

This example demonstrates:
1. how to bind mount a work directory into the container
2. how to copy files from the container to the host
3. how to run an interactive shell inside the container
4. how to launch a hybrid MPI+OpenMP container using the host MPI runtime
5. how to launch a CUDA container

[BigDFT](https://bigdft-suite.readthedocs.io/en/latest) is an electronic structure code targeting large molecular
systems with density functional theory. The program is written for heterogeneous computing
environments with support for both MPI, OpenMP and CUDA. This makes for a good test case
as a more advanced container application. All the following is based on the official
tutorial which can be found [here](https://ngc.nvidia.com/catalog/containers/hpc:bigdft).

A BigDFT Docker image with CUDA support is provided by the
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com/catalog)
and can be built using Singularity with the following command (here into a folder called
`$HOME/containers`, but this is arbitrary):
```
$ singularity pull --name $HOME/containers/bigdft-cuda.sif docker://nvcr.io/hpc/bigdft:cuda10-ubuntu1804-ompi4-mkl
```

```{warning}
Container images are typically a few GiB in size, so you might want to keep your
containers in a project storage area to avoid filling up your limited `$HOME` disk quota.
Also beware that pulled images are cached, by default under `$HOME/.singularity/cache`.
This means that if you pull the same image twice, it will be immediately available from
the cache without downloading/building, but it also means that it will consume disk space.
To avoid this you can either add `--disable-cache` to the `pull` command, change the cache
directory with the `SINGULARITY_CACHEDIR` environment variable, or clean up the cache
regularly with `singularity cache clean`.
```

### MPI + OpenMP example

The BigDFT container comes bundled with a couple of test cases that can be used to verify
that everything works correctly. We will start by extracting the necessary input files
for a test case called FeHyb which can be found in the `/docker/FeHyb` directory _inside_
the container (starting here with the non-GPU version):
```
$ mkdir $HOME/bigdft-test
$ singularity exec --bind $HOME/bigdft-test:/work-dir $HOME/containers/bigdft-cuda.sif /bin/bash -c "cp -r /docker/FeHyb/NOGPU /work-dir"
```
Here we first create a job directory for our test calculation on the host called `$HOME/bigdft-test`
and then bind mount this to a directory called `/work-dir` _inside_ the container. Then we execute
a bash command in the container to copy the example files from `/docker/FeHyb/NOGPU` into this
work directory, which is really the `$HOME/bigdft-test` directory on the host. You should now see a
`NOGPU` folder on the host file system with the following content:
```
$ ls $HOME/bigdft-test/NOGPU
input.yaml  log.ref.yaml  posinp.xyz  psppar.Fe  tols-BigDFT.yaml
```

```{note}
Container images are read-only, so it is not possible to copy things _into_ the container
or change it in any other way without sudo access on the host. This is why all container
_construction_ needs to be done on your local machine where you have such privileges, see
{ref}`guides <dev-guides>` for more info on building containers.
```

The next thing to do is to write a job script for the test calculation, we call it
`$HOME/bigdft-test/NOGPU/FeHyb.run`:
```bash
#!/bin/bash

#SBATCH --account=<myaccount>
#SBATCH --job-name=FeHyb-NOGPU
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00

# Need to provide a compatible MPI library on the host for launching the calculation
module load OpenMPI/4.0.3-GCC-9.3.0

# Run the bigdft command inside the bigdft-cuda.sif container
# We assume we are already in the folder containing the input.yaml
# file which means it is automatically mounted in the container
mpirun --bind-to none singularity exec $HOME/containers/bigdft-cuda.sif bigdft

exit 0
```

The `--bind-to none` option is necessary to avoid all OpenMP threads landing on the
same CPU core. Now set `<myaccount>` to something appropriate and launch the job
```
$ sbatch FeHyb.run
```
It should not take more than a minute to finish. After completion, the Slurm output
file should contain the following line (in addition to the usual Slurm statistics output):
```
 <BigDFT> log of the run will be written in logfile: ./log.yaml
```
To check that the calculation succeeded, we can inspect the `log.yaml` output file,
or we can run a test script provided by the container. First start an interactive shell
inside the container (you should run this command in the job directory containing the
`log.yaml` file so that it is automatically mounted in the container):
```
$ singularity shell $HOME/containers/bigdft-cuda.sif
```
Now you have stepped into the container and your shell prompt should have changed from `$`
to `Singularity>`. Now run the command:
```
Singularity> python /usr/local/bigdft/lib/python2.7/site-packages/fldiff_yaml.py -d log.yaml -r /docker/FeHyb/NOGPU/log.ref.yaml -t /docker/FeHyb/NOGPU/tols-BigDFT.yaml
```
which hopefully reports success, something like this:
```
---
Maximum discrepancy: 2.4000000000052625e-09
Maximum tolerance applied: 1.1e-07
Platform: c1-33
Seconds needed for the test: 11.92
Test succeeded: True
Remarks: !!map
  Report: {Document: 0, Elapsed Time (s): 11.923177122, Failed_checks: 0, Max_Diff: 2.4000000000052625e-09,
    Memory_leaks (B): 0, Missed_items: 0}
```
To exit the container, type `exit` or press `Ctrl-D`.

### CUDA example

We will now run the same example using the CUDA version of BigDFT. We again copy the
bundled input files from within the container, this time the `GPU` directory (see
example above for explanation of the commands):

```
singularity exec --bind $HOME/bigdft-test:/work-dir $HOME/containers/bigdft-cuda.sif /bin/bash -c "cp -r /docker/FeHyb/GPU /work-dir"
```
which should contain the following files:
```
$ ls $HOME/bigdft-test/GPU
input.yaml  posinp.xyz  psppar.Fe
```
In order to run this example correctly we need to ask for GPU resources in the job
script, here we call it `$HOME/bigdft-test/GPU/FeHyb.run`. We request a single CPU
core (`--ntasks=1`) with an associated GPU accelerator (`--gpus=1`). Also remember
to use the `accel` partition:
```bash
#!/bin/bash

#SBATCH --account=<myaccount>
#SBATCH --job-name=FeHyb-GPU
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --partition=accel
#SBATCH --time=00:10:00

# Run the bigdft command inside the bigdft-cuda.sif container
# We assume we are already in the folder containing the input.yaml
# file which means it is automatically mounted in the container
singularity exec --nv $HOME/containers/bigdft-cuda.sif bigdft

exit 0
```
With BigDFT, the CUDA request is handled through the input file, so we run the same
`bigdft` executable as before. There is an extra `--nv` option for the `singularity exec`
command though, which will make the container aware of the available NVIDIA hardware.
Set `<myaccount>` to something appropriate and launch the job
```
$ sbatch FeHyb.run
```
We can again check that the calculation completed successfully by shell-ing into the
container and running the diff script (note that we still compare against the `NOGPU`
reference as there is no specific GPU reference available in the container):
```
$ singularity shell $HOME/containers/bigdft-cuda.sif
Singularity> python /usr/local/bigdft/lib/python2.7/site-packages/fldiff_yaml.py -d log.yaml -r /docker/FeHyb/NOGPU/log.ref.yaml -t /docker/FeHyb/NOGPU/tols-BigDFT.yaml
---
Maximum discrepancy: 2.4000000000052625e-09
Maximum tolerance applied: 1.1e-07
Platform: c7-6
Seconds needed for the test: 11.85
Test succeeded: True
Remarks: !!map
  Report: {Document: 0, Elapsed Time (s): 11.84816281, Failed_checks: 0, Max_Diff: 2.4000000000052625e-09,
    Memory_leaks (B): 0, Missed_items: 0}
```
As we can see from the timings, this small test case run more or less equally fast (11-12 sec)
on a single GPU as on 2x10 CPU cores. For comparison, the same example takes about 90 sec
to complete on a single CPU core.
