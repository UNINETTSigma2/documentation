(bigdft-cuda-example)=
# BigDFT with MPI and CUDA

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
```console
[SAGA]$ singularity pull --name $HOME/containers/bigdft-cuda.sif docker://nvcr.io/hpc/bigdft:cuda10-ubuntu1804-ompi4-mkl
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

## MPI + OpenMP version

The BigDFT container comes bundled with a couple of test cases that can be used to verify
that everything works correctly. We will start by extracting the necessary input files
for a test case called FeHyb which can be found in the `/docker/FeHyb` directory _inside_
the container (starting here with the non-GPU version):
```console
[SAGA]$ mkdir $HOME/bigdft-test
[SAGA]$ singularity exec --bind $HOME/bigdft-test:/work-dir $HOME/containers/bigdft-cuda.sif /bin/bash -c "cp -r /docker/FeHyb/NOGPU /work-dir"
```
Here we first create a job directory for our test calculation on the host called `$HOME/bigdft-test`
and then bind mount this to a directory called `/work-dir` _inside_ the container. Then we execute
a bash command in the container to copy the example files from `/docker/FeHyb/NOGPU` into this
work directory, which is really the `$HOME/bigdft-test` directory on the host. You should now see a
`NOGPU` folder on the host file system with the following content:
```console
[SAGA]$ ls $HOME/bigdft-test/NOGPU
input.yaml  log.ref.yaml  posinp.xyz  psppar.Fe  tols-BigDFT.yaml
```

```{note}
Container images are read-only, so it is not possible to copy things _into_ the container
or change it in any other way without sudo access on the host. This is why all container
_construction_ needs to be done on your local machine where you have such privileges, see [guides](code_development) for more info on building containers.
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
# file and we bind the current directory into the container
mpirun --bind-to none singularity exec --bind $PWD $HOME/containers/bigdft-cuda.sif bigdft

exit 0
```

The `--bind-to none` option is necessary to avoid all OpenMP threads landing on the
same CPU core. Now set `<myaccount>` to something appropriate and launch the job
```console
[SAGA]$ sbatch FeHyb.run
```
It should not take more than a minute to finish. After completion, the Slurm output
file should contain the following line (in addition to the usual Slurm statistics output):
```console
 <BigDFT> log of the run will be written in logfile: ./log.yaml
```
To check that the calculation succeeded, we can inspect the `log.yaml` output file,
or we can run a test script provided by the container. First start an interactive shell
inside the container (you should run this command in the job directory containing the
`log.yaml` file so that we find it when we bind to the current directory):
```console
[SAGA]$ singularity shell --bind $PWD $HOME/containers/bigdft-cuda.sif
```
Now you have stepped into the container and your shell prompt should have changed from `$`
to `Singularity>`. Now run the command:
```console
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

## CUDA version

We will now run the same example using the CUDA version of BigDFT. We again copy the
bundled input files from within the container, this time the `GPU` directory (see
example above for explanation of the commands):

```console
[SAGA]$ singularity exec --bind $HOME/bigdft-test:/work-dir $HOME/containers/bigdft-cuda.sif /bin/bash -c "cp -r /docker/FeHyb/GPU /work-dir"
```
which should contain the following files:
```console
[SAGA]$ ls $HOME/bigdft-test/GPU
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
# file and we bind the current directory into the container
singularity exec --nv --bind $PWD $HOME/containers/bigdft-cuda.sif bigdft

exit 0
```
With BigDFT, the CUDA request is handled through the input file, so we run the same
`bigdft` executable as before. There is an extra `--nv` option for the `singularity exec`
command though, which will make the container aware of the available NVIDIA hardware.
Set `<myaccount>` to something appropriate and launch the job
```console
[SAGA]$ sbatch FeHyb.run
```
We can again check that the calculation completed successfully by shell-ing into the
container and running the diff script (note that we still compare against the `NOGPU`
reference as there is no specific GPU reference available in the container):
```console
[SAGA]$ singularity shell --bind $PWD $HOME/containers/bigdft-cuda.sif
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

