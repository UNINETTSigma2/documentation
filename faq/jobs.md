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

### Job start gets postponed
Note: This is specific for *Fram*.  If the estimated job start gets
postponed again and again, one reason might be that the queue system
cannot find idle nodes within the same "island" of Fram.  The network
on Fram is divided into four "islands", and the network is faster
within the islands than between them.  The queue system will by
default delay a job with up to 7 days in order to be able to start it
on a single island.  This is most likely to happen for jobs asking for
many nodes.  [It is possible to override this.](../jobs/framjobplacement.md)

## Things to avoid

Things you should not do or use when running jobs on the HPC clusters.

### `#SBATCH --hint=nomultithread`
Do not use this, at least not with Intel MPI jobs.  If you do, the result is
that all the tasks (ranks) will be bound to the first CPU core on each compute
node.
