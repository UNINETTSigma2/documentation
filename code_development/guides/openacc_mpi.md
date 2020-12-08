# Combining MPI and OpenACC
A lot of existing HPC code is already set up to take advantage of multi-core and
cluster compute resources through `MPI`. When translating a codebase to OpenACC
we can take advantage of this existing infrastructure by giving each `rank` its
own GPU to achieve multi-GPU compute. This technique is most likely the easiest
path to utilizing multi-GPU for existing and new projects working with OpenACC.

```{note}
The alternative to combining `MPI` and OpenACC is to divide the work into
blocks, as we show in the [asynchronous and multi-GPU
guide](./async_openacc.md), however, with out combining such a technique with
`MPI`, sharing is limited to a single node.
```

```{tip}
For a summary of available directives we have used [this reference
guide.](https://www.openacc.org/sites/default/files/inline-files/API%20Guide%202.7.pdf)
```

## Introduction
This guide will assume some familiarity with `MPI` and an [introductory
level](./openacc.md) of knowledge about OpenACC.

After reading this guide you should be familiar with the following concepts
 - How to take an existing `MPI` application and add OpenACC
    - How to assign the correct GPU based on `rank` and nodes
    - How to share data between `CPU`, `GPU` and other `rank`s

For this guide we will solve the 1 dimensional wave equation, shown below with
`MPI` task sharing.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_mpi.c
   :language: c
```

```{eval-rst}
:download:`wave_mpi.c <./openacc_mpi/wave_mpi.c>`
```

To compile this on Saga we will load `NVHPC` and compile with the built-in `MPI`
compiler.

```bash
$ module load NVHPC/20.7
$ mpicc -g -fast -o mpi wave_mpi.c
```

```{warning}
It might be necessary to export the path to `mpicc` for the above to work. Run
the following to set it up correctly: `export
PATH=$PATH:$NVHPC/Linux_x86_64/20.7/comm_libs/mpi/bin`
```

To run this with multiple `rank`s, e.g. split the work over `4` processes, use
the following command
```bash
$ srun --ntasks=4 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./mpi 1000000
```

## Introducing OpenACC
When starting the transition of an `MPI` enabled application, like the above, it
is imperative to reduce complexity so as to not get overwhelmed by the
transition. We will therefore introduce OpenACC to the program by running it as
one process and once that is done add the necessary setup for multi-GPU
utilization.

```{tip}
Try to follow the next sections by implementing them yourself before you see our
solution. This will increase your confidence with OpenACC.
```

### Adding `parallel loop` directives
There are several places where we could put directives in this code, however, to
keep this manageable we will focus on the main computational area of the code.
Lets therefore start with the following three loops.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_mpi.c
   :language: c
   :lines: 179-222
   :emphasize-lines: 6-11, 14-18, 40-43
```

Looking at the three loops we can see that every iteration is independent of
every other iteration and it is thus safe to add `#pragma parallel loop` before
each loop.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_loop.c
   :language: c
   :lines: 179-225
   :emphasize-lines: 6, 15, 42
```

```{eval-rst}
:download:`wave_loop.c <./openacc_mpi/wave_loop.c>`
```

To compile this on Saga we use the same command as above, adding `-acc` and
`-Minfo=accel`

```bash
$ mpicc -g -fast -acc -Minfo=accel -o acc wave_loop.c
```

To test the above code use `srun` as above, but do not ask for multiple tasks.
We also need to request GPU resources with `--partition=accel` and
`--gres=gpu:1`.

```bash
$ srun --ntasks=1 --partition=accel --gres=gpu:1 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./acc 1000000
```

The code runs on the GPU, but it is not particularly fast. The reason for this
is that we are now continually copying memory in and out of the GPU. If we look
at the main computation we can see that, apart from sharing two elements of the
array with the other `rank`s, we don't need to work on the data on the `CPU`.

### Improving data locality
To improve data locality we need to know which pieces of information, e.g. which
arrays, are important to have on `GPU` and those we don't need to transfer.
Taking a step back and thinking about the code we can see that `wave0` and
`wave2` are only used for scratch space while the end result ends up in `wave1`.
In addition we can see that this holds true, except for variable sharing with
`MPI` - which we will come back to below, for the whole `steps` loop.

Lets add a `#pragma acc data` directive above the `steps` loop so that data is
contained on the `GPU` for the whole computation.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_data.c
   :language: c
   :lines: 178-200
   :emphasize-lines: 2-3
```

```{eval-rst}
:download:`wave_data.c <./openacc_mpi/wave_data.c>`
```

Compile and run to see if we get any improvements.

```bash
$ mpicc -g -fast -acc -Minfo=accel -o acc wave_data.c
$ srun --ntasks=1 --partition=accel --gres=gpu:1 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./acc 1000000
```

The above runs quite a bit faster, but we have one problem now, the output is
wrong. This is due to the fact that we are now not sharing any data between the
`GPU` and the `CPU`. To improve this we will introduce the `#pragma acc update`
directive.

### Transfer between `GPU <=> CPU`
As we just saw, we are missing some data on the `CPU` to initiate the `MPI`
transfer with. To remedy this we can add the `#pragma acc update` directive.
This directive tells the compiler to transfer data to or from `GPU` without an
associated block.

First we will copy data from the `GPU` back to the `CPU` so that our `MPI`
transfer can proceed. In the code below notice that we have added `acc update
self(...)`. `self` in this context means that we want to transfer from `GPU` to
`CPU`.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc.c
   :language: c
   :lines: 197-226
   :emphasize-lines: 10-11
```

The `MPI` transfer will transmit the correct data now, which is good, but we
still have a problem in our code. After the `MPI` transfer the points we
received from the other `rank`s are not updated on the `GPU`. To fix this we can
add the same `acc update` directive, but change the direction of the transfer.
To do this we change `self` with `device` as follows.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc.c
   :language: c
   :lines: 224-236
   :emphasize-lines: 5-6
```

We have made a few more improvements to the overall code to most fairly compare
with pure `MPI`. See the `wave_acc.c` file below for additional improvements.

```{eval-rst}
:download:`wave_acc.c <./openacc_mpi/wave_acc.c>`
```

Compile and run with the following, as usual.

```bash
$ mpicc -g -fast -acc -Minfo=accel -o acc wave_acc.c
$ srun --ntasks=1 --partition=accel --gres=gpu:1 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./acc 1000000
```

## Splitting the work over multiple `GPU`s
We are almost done with our transition to OpenACC, however, what happens if we
launch the above `wave_acc.c` with two `rank`s on the same node. From the
perspective of the batch system we will be allocated two `GPU`s, two `CPU` cores
and a total `512M+512M` of memory. However, our two `MPI` processes will utilize
the default `GPU` and since they are running on the same node, that will be the
same `GPU`.

To do this we will read in our local `rank` which is exported under OpenMPI as
`OMPI_COMM_WORLD_LOCAL_RANK`. Then we can use this to get the correct index to
the `GPU` to use. We need to add `#include <openacc.h>` so that we can access
the Nvidia runtime and have access to `acc_set_device_num()` which we can use to
assign a unique `GPU` to each `MPI` process.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc_mpi.c
   :language: c
   :lines: 91-122
   :emphasize-lines: 21-33
```

```{eval-rst}
:download:`wave_acc_mpi.c <./openacc_mpi/wave_acc_mpi.c>`
```

We will compile this as before, but now we can run with arbitrary number of
processes!

```bash
$ mpicc -g -fast -acc -Minfo=accel -o acc wave_acc_mpi.c
$ srun --ntasks=2 --partition=accel --gres=gpu:1 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./acc 1000000
```

## Summary
We have shown how to take an existing `MPI` application and add OpenACC to
utilize multi-GPU resources. To accomplish this, we added directives to move
small amounts of data back and fourth between the `GPU` and `CPU` so that we
could continue to exchange data with neighboring `MPI` `rank`s.
