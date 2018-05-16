# Running MPI Applications

On Fram users have access to two MPI implementations:

* OpenMPI is provided by the `foss` module, e.g., `module load
foss/2017a` 
* Intel MPI environment is provided by the `intel` module, e.g.,
`module load intel/2017a` 

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

## OpenMPI
On systems with Mellanox InfiniBand, OpenMPI is the implementation recommended by Mellanox due to it's support
for the [HPCX communication libraries](http://www.mellanox.com/page/products_dyn?product_family=189&mtag=hpc-x).

### `srun`

With OpenMPI, `srun` is the preferred way to start MPI programs due to good integration with the Slurm scheduler environment:

```
srun -n <num ranks> /path/to/openmpi_app
```

Executed as above, `srun` uses SLURM's default binding and mapping algorithms (currently
`--cpu_bind=cores`), [which can be changed](https://slurm.schedmd.com/srun.html) using either command-line
parameters, or environment variables. Parameters specific to OpenMPI can be set using [environment variables](https://www.open-mpi.org/faq/?category=tuning#setting-mca-params).

Note that with `srun` it is necessary to explicitly specify the number of ranks to start with the `-n` parameter. Otherwise
SLURM will by default start one rank per compute node.

### `mpirun`

For those familiar with the OpenMPI tools, MPI applications can also be started using the `mpirun` command:

```
mpirun /path/to/openmpi_app
```

By default, `mpirun` binds ranks to cores, and maps them by socket. Please refer to the
[documentation](https://www.open-mpi.org/doc/v2.1/man1/mpirun.1.php)
if you need to change those settings. Note that `-report-bindings` is
a very useful option if you want to inspect the individual MPI ranks
to see on which nodes, and on which CPU cores they run.


## Intel MPI

### `mpirun`
At this moment, for performance reasons `mpirun` is the preferred way
to start applications that use Intel MPI:

```
mpirun /path/to/intelmpi_app
```

In the above, `app` is subject to `mpirun`'s
internal mapping and binding algorithms. Intel's `mpirun` uses it's own default
binding settings, which can be modified either by [command line
parameters](https://software.intel.com/en-us/node/589999), or by
[environment
variables](https://software.intel.com/en-us/mpi-developer-reference-windows-environment-variables-for-process-pinning).
Special care must be taken when running hybrid MPI-OpenMP cores. If
this is your case, please refer to the documentation regarding
[Interoperability between MPI and OpenMP](https://software.intel.com/en-us/mpi-developer-reference-windows-interoperability-with-openmp-api).

### `srun`
With `srun`, Intel MPI applications cab be started as follows:

```
srun -n <num ranks> --mpi=pmi2 /path/to/app
```

Note that you must explicitly specify `--mpi=pmi2`.

We have observed that in the current setup some applications compiled against Intel MPI and executed
with `srun` achieve inferior performance compared to the same
code executed with `mpirun`. Until this is resolved, to start applicationswe suggest using `mpirun`.

## Final remarks

Note that when executing `mpirun` from within a SLURM allocation there
is no need to provide neither the number of MPI ranks (`-np`), nor the host file
(`-hostfile`): those are obtained automatically by `mpirun`. When using `srun` one has to explicitly add
the `-n <num ranks>` parameter, otherwise SLURM will by default start one rank per compute node.
