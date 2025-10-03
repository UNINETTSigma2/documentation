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
