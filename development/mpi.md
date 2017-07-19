
## MPI

Currently, the supported Message Passing Interface (MPI) implementation on Fram is Intel MPI, and is
accessed by loading the Intel MPI module:

	module load intel/2017a

Intel MPI jobs can be launced in several ways. The recommended way is to use `srun` inside the job script file:

	module load intel/2017a

	srun MyProgram

`srun` will start the number of tasks (ranks) on each node that has been
specified by `--ntasks-per-node` or `--ntasks`, and defaults to 1 task per
node.

It is also possible to use `mpirun` in jobs, but this is currently not
recommended, and one has to first to unset the environment variable
`I_MPI_PMI_LIBRARY` (it is set in job scripts), otherwise `mpirun` will fail.
Using `mpiexec` in jobs is currently not supported.
