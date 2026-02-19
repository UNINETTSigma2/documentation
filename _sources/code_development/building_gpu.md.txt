(building-gpu)=
# Building GPU software

The login nodes on Betzy and Saga currently do not allow to compile software for the GPUs
as the cuda driver is not installed.
In order to compile the GPU software one needs an interactive session on a GPU node.
If the GPU is not needed, one can ask for a CPU-only allocation, e.g.:

```
salloc --nodes=1 --time=00:30:00 --partition=<accel|a100> --mem-per-cpu=8G --account=<...>
```

or, if GPUs are required, e.g., for testing purposes:

```
salloc --nodes=1 --time=00:30:00 --partition=<accel|a100> --mem-per-cpu=8G --account=<...> --gpus=1
```

## Saga

There are two types of GPU nodes on Saga, located in two distinct SLURM partitions:

* Intel CPU  with 4X Tesla P100, 16GB, `--partition=accel`
* AMD CPUs with 4XA100, 80GB, `--partition=a100`

These are different architectures. By default, Saga loads the Intel software environment,
If you want to run/compile software for the nodes with the AMD CPUs and A100 GPUS 
you need to get an allocation on the `a100` partition. Then, inside the allocation,
or inside your job script, switch the module environment

```
module --force swap StdEnv Zen2Env

```  
Note that installed modules can vary between the two node types.

## Olivia

Olivia has a unique architecture where the **accelerator/GPU nodes use ARM-based CPUs** (NVIDIA Grace CPUs), while the **login nodes use x86-64 architecture**. This means that software built on the login nodes will **not** run on the GPU partition.

```{warning}
**Important**: To build software for the GPU partition on Olivia, you **must** build it directly on the GPU nodes, not on the login nodes.
```

To compile software for the GPU nodes, you can either use an interactive job or a job script. We recommend building on top of the `NRIS/GPU` software stack.

For an interactive session on the accelerator partition (without requesting a GPU):

```
salloc --partition=accel --time=0:30:00 --ntasks=1 --cpus-per-task=8 --mem=16G --gpus=0 --account=<...>
```

Once in the interactive session, load the `NRIS/GPU` stack and proceed with building your software:

```
module load NRIS/GPU
```

Alternatively, you can use a job script to build your software. This ensures that binaries are compiled for the ARM architecture used by the GPU nodes.
