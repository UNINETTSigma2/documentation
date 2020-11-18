(running-mpi-applications)=

# Running MPI Applications

On Fram and Saga users have access to two MPI implementations:

* OpenMPI is provided by the foss - and iomkl toolchains; and may also
  be loaded directly. For available versions, type `module avail
  OpenMPI/`**(note the slash)**. Normal way of loading is through the
  `foss`-toolchain module, e.g. `module load foss/2018a`
* Intel MPI environment is provided by the intel-toolchain and may
  also be loaded directly. For available versions, type `module avail
  impi/`. Normal way of loading is through the `intel`-toolchain
  module, e.g.  `module load intel/2018a`

**Also note that quite a few scientific packages is set up in such a
way that all necessary software are loaded as a part of the software
module in question. Do not load toolchains and/or mpi modules
explicitely unless absolutely sure of the need for it!!!**

Slurm is used as the [queue system](queu_system.md), and the native
way to start MPI applications with Slurm is to use the
[`srun`](https://slurm.schedmd.com/srun.html) command. On the other
hand, both MPI implementations provide their own mechanisms to start
application in the form of the `mpirun` command.

One of the most important factors when running large MPI jobs is
mapping of the MPI ranks to compute nodes, and *binding* (or
*pinning*) them to CPU cores. Neglecting to do that, or doing that in
an suboptimal way can severely affect performance. In this regard
there are some differences when it comes to running applications
compiled against the two supported MPI environments.

**Also note that the choice of MPI should be based on which MPI the
code is compiled with support for.** So if `module list` give you a
`OpenMPI/`-reading, you should focus on the OpenMPI part beneath, if
given a `impi/`-reading focus on the Intel MPI part

## OpenMPI

On systems with Mellanox InfiniBand, OpenMPI is the implementation
recommended by Mellanox due to it's support for the [HPCX
communication
libraries](https://docs.mellanox.com/category/hpcx).

### `srun`

With OpenMPI, `srun` is the preferred way to start MPI programs due to
good integration with the Slurm scheduler environment:

```
srun /path/to/MySoftWare_exec
```

Executed as above, `srun` uses Slurm's default binding and mapping
algorithms (currently `--cpu_bind=cores`), [which can be
changed](https://slurm.schedmd.com/srun.html) using either
command-line parameters, or environment variables. Parameters specific
to OpenMPI can be set using [environment
variables](https://www.open-mpi.org/faq/?category=tuning#setting-mca-params).

In the above scenario `srun` uses the PMI2 interface to launch the MPI
ranks on the compute nodes, and to exchange the InfiniBand address information between
the ranks. For large jobs the startup might be faster using OpenMPI's PMIx method:

```
srun --mpi=pmix /path/to/MySoftWare_exec
```

The startup time might be improved further using the OpenMPI MCA
`pmix_base_async_modex` argument (see below). With `srun` this needs to be
set using an environment variable.

### `mpirun`

```{warning}
**On Saga use srun, not mpirun**

mpirun can get the number of tasks wrong and also lead to wrong task
placement. We don't fully understand why this happens. When using srun
instead of mpirun or mpiexec, we observe correct task placement on Saga.
```

For those familiar with the OpenMPI tools, MPI applications can also
be started using the `mpirun` command:

```
mpirun /path/to/MySoftWare_exec
```

By default, `mpirun` binds ranks to cores, and maps them by
socket. Please refer to the
[documentation](https://www.open-mpi.org/doc/v2.1/man1/mpirun.1.php)
if you need to change those settings. Note that `-report-bindings` is
a very useful option if you want to inspect the individual MPI ranks
to see on which nodes, and on which CPU cores they run.

When launching large jobs with sparse communication patterns
(neighbor to neighbor, local communication) the startup time will be improved
by using the following command line argument:

```
mpirun -mca pmix_base_async_modex 1 ...
```
In the method above the address information will be exchanged between the ranks on a
need-to-know basis, i.e., at first data exchange between two ranks, instead of an all to all communication
step at program startup. Applications with dense communication patterns (peer to peer exchanges
with all ranks) will likely experience a slowdown.


## Intel MPI

### `mpirun`

```{warning}
**On Saga use srun, not mpirun**

mpirun can get the number of tasks wrong and also lead to wrong task
placement. We don't fully understand why this happens. When using srun
instead of mpirun or mpiexec, we observe correct task placement on Saga.
```

At this moment, for performance reasons `mpirun` is the preferred way
to start applications that use Intel MPI:

```
mpirun /path/to/MySoftWare_exec
```

In the above, `MySoftWare_exec` is subject to `mpirun`'s internal
mapping and binding algorithms. Intel's `mpirun` uses it's own default
binding settings, which can be modified either by [command line
parameters](https://software.intel.com/en-us/node/589999), or by
[environment
variables](https://software.intel.com/en-us/mpi-developer-reference-windows-environment-variables-for-process-pinning).
Special care must be taken when running hybrid MPI-OpenMP cores. If
this is your case, please refer to the documentation regarding
[Interoperability between MPI and
OpenMP](https://software.intel.com/en-us/mpi-developer-reference-windows-interoperability-with-openmp-api).

### `srun`

With `srun`, Intel MPI applications can be started as follows:

```
srun /path/to/MySoftWare_exec
```

We have observed that in the current setup some applications compiled
against Intel MPI and executed with `srun` achieve inferior
performance compared to the same code executed with `mpirun`. Until
this is resolved, we suggest using `mpirun` to start applications.

## Final remarks

Note that when executing `mpirun` from within a Slurm allocation there
is no need to provide neither the number of MPI ranks (`-np`), nor the
host file (`-hostfile`): those are obtained automatically by
`mpirun`.  This is also be the case with `srun`.
