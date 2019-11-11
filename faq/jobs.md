# Running jobs

## FAQ

Frequently asked questions related to running jobs in the queue system on the
HPC clusters (currently: Fram and Saga).

### "srun: Warning: can't honor --ntasks-per-node set to _X_ which doesn't match the requested tasks _Y_ with the number of requested nodes _Y_. Ignoring --ntasks-per-node."
This warning appears when using the `mpirun` command with Intel MPI and
specifying `--ntasks-per-node` for jobs in the `normal` partition on Fram.  As
far as we have seen, the job does *not* ignore the `--ntasks-per-node`, and
will run the specified number of processes per node.  You can test it with,
e.g., `mpirun hostname`.  Please let us know if you have an example where
`--ntasks-per-node` is *not* honored!

So, if you get this when using `mpirun` with Intel MPI, our recommendation is
currently that the warning can be ignored.

## Things to avoid

Things you should not do or use when running jobs on the HPC clusters.

### `#SBATCH --hint=nomultithread`
Do not use this, at least not with Intel MPI jobs.  If you do, the result is
that all the tasks (ranks) will be bound to the first CPU core on each compute
node.
