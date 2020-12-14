```{index} MPI; Combining MPI and OpenACC, OpenACC; Combining MPI and OpenACC, GPU; Combining MPI and OpenACC, Nvidia Nsight; Combining MPI and OpenACC, Multi-GPU; Combining MPI and OpenACC
```
# Combining MPI and OpenACC
A lot of existing HPC code is already set up to take advantage of multi-core and
cluster compute resources through `MPI`. When translating a codebase to OpenACC
we can take advantage of this existing infrastructure by giving each `rank` its
own `GPU` to achieve multi-GPU compute. This technique is most likely the easiest
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
    - How to go from an initial `CPU`-only implementation to gradually adding
        OpenACC directives
    - How to share data between `CPU`, `GPU` and other `rank`s
    - How to assign the correct GPU based on `rank` and nodes
 - How to profile a combined `MPI` and OpenACC application

---

For this guide we will solve the 1 dimensional wave equation, shown below with
`MPI` task sharing.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_mpi.c
   :language: c
```

```{eval-rst}
:download:`wave_mpi.c <./openacc_mpi/wave_mpi.c>`
```

To compile this on Saga we will load `OpenMPI` and compile with the built-in `MPI`
compiler.

```bash
$ module load OpenMPI/4.0.3-PGI-20.4-GCC-9.3.0
$ mpicc -g -fast -o mpi wave_mpi.c
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
one process and once we are done with the OpenACC translation, add the necessary
setup for multi-GPU utilization.

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

#### Checking with Nsight
To see the problem visually, we can use [Nvidia
Nsight](https://developer.nvidia.com/nsight-systems) to profile the application.
We will simply change the invocation of `time` with `nsys` as follows.

```bash
$ srun --ntasks=1 --partition=accel --gres=gpu:1 --account=<your project number> --time=02:00 --mem-per-cpu=512M nsys profile -t cuda,openacc,osrt -o openacc_mpi_tutorial ./acc 1000000
```

This will create an `openacc_mpi_tutorial.qdrep` profile that we can download to
our local machine and view in Nsight Systems.

![Screenshot of Nvidia Nsight showing kernel and memory usage of our
`wave_loop.c` program](./openacc_mpi/wave_loop_profile.png)

```{note}
Right click on the image and select `View Image` to see larger version.
```

```{tip}
If you have not used Nsight before we have an [introductory tutorial
available](./openacc.md).
```

As we can see, in the image above, the `Kernels` to `Memory` ratio is quite one
sided, confirming our suspicion that we are spending too much time transferring
data, and not enough time computing.


### Improving data locality
To improve data locality we need to know which pieces of information, e.g. which
arrays, are important to have on `GPU` and those we don't need to transfer.
Taking a step back and thinking about the code we can see that `wave0` and
`wave2` are only used for scratch space while the end result ends up in `wave1`.
In addition we can see that this holds true, except for variable sharing with
`MPI` - which we will come back to below, for the whole `steps` loop.

Lets add a `#pragma acc data` directive above the `steps` loop so that data is
contained on the `GPU` for the whole computation. Since we have some data in
`wave0` we will mark it as `copyin`, we need the data in `wave1` after the loop
as well so we mark it as `copy` and `wave2` we can just create on the `GPU`
since it is only used as scratch in the loop.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_data.c
   :language: c
   :lines: 178-200
   :emphasize-lines: 2-3
```

```{eval-rst}
:download:`wave_data.c <./openacc_mpi/wave_data.c>`
```

```{note}
We had to remove the `check_mpi` calls in the region covered by the `#pragma acc
data` since it is not allowed to exit out of a compute region with `return`.
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

````{tip}
Use `nsys` and Nvidia Nsight to convince
yourself that the above stated improvement actually takes place.
```{eval-rst}
:download:`Nsight updated screenshot <./openacc_mpi/wave_data_profile.png>`
```
````

### There and back again - `GPU <=> CPU`
As we just saw, we are missing some data on the `CPU` to initiate the `MPI`
transfer with. To remedy this we will add the `#pragma acc update` directive.
This directive tells the compiler to transfer data to or from `GPU` without an
associated block.

First we will copy data from the `GPU` back to the `CPU` so that our `MPI`
transfer can proceed. In the code below notice that we have added `acc update
self(...)`. `self` in this context means that we want to transfer from `GPU` to
`CPU`.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc.c
   :language: c
   :lines: 197-222
   :emphasize-lines: 10-11
```

The `MPI` transfer will transmit the correct data, which is good, but we still
have a problem in our code. After the `MPI` transfer the points we received from
the other `rank`s are not updated on the `GPU`. To fix this we can add the same
`acc update` directive, but change the direction of the transfer.  To do this we
change `self` with `device` as follows.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc.c
   :language: c
   :lines: 220-232
   :emphasize-lines: 5-6
```

We have made a few more improvements to the overall code to more fairly compare
with the pure `MPI` solution. See the `wave_acc.c` file below for additional
improvements.

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
and a total of `512M+512M` memory. However, our two `MPI` processes do not
specify the `GPU` to use and will utilize the default `GPU`. Since they are
running on the same node, that will likely be the same `GPU`.

To fix this we will read in our local `rank`, which is exported under OpenMPI as
`OMPI_COMM_WORLD_LOCAL_RANK`, then we can use this to get the correct index to
the `GPU` to use. We need to add `#include <openacc.h>` so that we can access
the Nvidia runtime and have access to `acc_set_device_num()` which we can use to
assign a unique `GPU` to each `MPI` process.

```{eval-rst}
.. literalinclude:: openacc_mpi/wave_acc_mpi.c
   :language: c
   :lines: 91-121
   :emphasize-lines: 21-32
```

```{eval-rst}
:download:`wave_acc_mpi.c <./openacc_mpi/wave_acc_mpi.c>`
```

We will compile this as before, but now we can run with arbitrary number of
processes!

```{warning}
When using `--gres=gpu:X` we need to request the total number of `GPU`s that we
expect each node to use. So if we ask for `--ntasks=2` we need to request two
`GPU`s with `--gres=gpu:2`. This will soon be remedied and can instead be
requested with `--gpus-per-task`.
```

```bash
$ mpicc -g -fast -acc -Minfo=accel -o acc wave_acc_mpi.c
$ srun --ntasks=2 --partition=accel --gres=gpu:2 --account=<your project number> --time=02:00 --mem-per-cpu=512M time ./acc 1000000
```

## Summary
We have shown how to take an existing `MPI` application and add OpenACC to
utilize multi-GPU resources. To accomplish this, we added directives to move
compute from `CPU` to `GPU`. To enable synchronization we also added directives
to move small amounts of data back and fourth between the `GPU` and `CPU` so
that we could continue to exchange data with neighboring `MPI` `rank`s.

### Speedup
Below is the runtime, as measured with `time -p ./executable` (extracting
`real`), of each version.  The code was run with `1200000` points to solve.

| Version | Time in seconds | Speedup |
| ------- | --------------- | ------- |
| `MPI` `--ntasks=1` | `14.29`| N/A |
| `MPI` `--ntasks=12`\* | `2.16` | `6.61x` |
| `MPI` `--ntasks=2` + OpenMP `--cpus-per-task=6`\* | `2.11` | `1.02x` |
| `MPI` `--ntasks=2` + OpenACC | `2.79` | `0.76x` |
| OpenACC\*\* | `2.17` | `1.28x` |
**\*** To keep the comparison as fair as possible we compare the `CPU` resources
that would be the equivalent to [the billing resources of 2 `GPU`s on
Saga](../../jobs/projects_accounting.md).

**\*\*** OpenACC implementation on a single `GPU` further optimized when no
`MPI` sharing is necessary.

### Scaling
To illustrate the benefit of combining OpenACC with `MPI` we have, in the image
below, compared three different versions of the solver on increasingly larger
input.

![Scaling of different inputs](./openacc_mpi/wave_scaling.svg)

From the figure we can see that on the example input, `1200000`, `MPI` combined
with OpenMP is the quickest. However, as we scale the input size this `CPU`
version becomes slower compared to the `GPU`. The figure also illustrates the
advantage of a single `GPU`. We recommended, if possible, to use one `GPU` when
starting the transition to OpenACC and if the data size is larger than a single
`GPU` continue with `MPI`. If the application already utilizes `MPI` then we
recommend that, when using OpenACC, each `rank` is given more work than in the
pure `CPU` implementation.
