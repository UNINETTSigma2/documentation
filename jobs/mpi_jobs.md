# Running MPI Applications

On Fram users have access to two MPI implementations:

* Intel MPI environment is provided by the `intel` module, e.g.,
`module load intel/2017a` 
* OpenMPI is provided by the `foss` module, e.g., `module load
foss/2017a` 

SLURM is used as the [queuing system and the resource
manager](jobscripts.md), and the native way to start MPI applications 
with SLURM is to use the [`srun`](https://slurm.schedmd.com/srun.html)
command. On the other hand, both MPI implementations provide their own
mechanisms to start application in the form of the `mpirun`
command.

One of the most important factors when running large MPI jobs is
mapping of the MPI ranks to compute nodes, and *binding* (or
*pinning*) them to CPU cores. Neglecting to do that, or doing that in
an inoptimal way can severely affect the performance. In this regard
there are some differences when it comes to running applications
compiled against the two supported MPI environments.

## Intel MPI

### `mpirun`
At this moment, for performance reasons `mpirun` is the preferred way
to start applicatoins that use Intel MPI. However, due to the fact that
SLURM and IMPI use conflicting  *process-to-core* binding procedures,
before one can use `mpirun` it is ***crucial*** to turn off SLURM's
binding:

```
export SLURM_CPU_BIND=none
mpirun /path/to/app
```

If not done, all MPI ranks will be executed on one CPU core only. This
***has to be done manually*** by the user in the job script, before 
executing `mpirun`. As a result, `app` is subject to `mpirun`'s
internal mapping and binding algorithms. Intel's `mpirun` uses default
binding settings, which can be modified either by [command line
parameters](https://software.intel.com/en-us/node/589999), or by
[environment
variables](https://software.intel.com/en-us/mpi-developer-reference-windows-environment-variables-for-process-pinning).
Special care must be taken when running hybrid MPI-OpenMP cores. If
this is your case, please refer to the documentation regarding
[Interoperability between MPI and OpenMP](https://software.intel.com/en-us/mpi-developer-reference-windows-interoperability-with-openmp-api).

### `srun`
With `srun`, Intel MPI applications have to be started as follows:

```
srun --mpi=pmi2 /path/to/app
```

`srun` uses SLURM's default binding and mapping algorithms (currently
`--cpu_bind=cores`), [which can be changed](https://slurm.schedmd.com/srun.html) using either command-line
parameters, or envirronment variables.

We have observed that in the current setup some applications executed
with `srun` achieve slightly inferior performance compared to the same
code executed with `mpirun`. Until this is resolved we suggest to use `mpirun`.


## Open MPI

### `mpirun`

For those familiar with the OpenMPI tools, `mpirun` can be used
without any further changes to the environment:

```
mpirun /path/to/app
```

By default, `mpirun` binds ranks to cores, and maps them by
socket. Please refer to the
[documentation](https://www.open-mpi.org/doc/v2.1/man1/mpirun.1.php)
if you need to change those settings. Note that `-report-bindings` is
a very useful option if you want to inspect the individual MPI ranks
to see on which nodes, and on which CPU cores they run.

### `srun`

With OpenMPI, `srun` should be used as follows:

```
srun --mpi=pmix /path/to/app
```

For the time being, because of lower performance of applicatoins
started with `srun`, we advice to use the `mpirun` command instead.

## Final remarks

Note that when executing `mpirun` from within a SLURM allocation there
is no need to provide neither the number of MPI ranks (`-np`), nor the host file
(`-hostfile`): those are obtained automatically by `mpirun`.
