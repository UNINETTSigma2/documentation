---
orphan: true
---

(running-job-steps-parallel)=

# Packaging smaller parallel jobs into one large

There are several ways to package smaller parallel jobs into one large
parallel job. *The preferred way is to use {ref}`array-jobs`.*
Here we want to present a more pedestrian alternative which can give a
lot of flexibility, but can also be a little more complicated to get right.

Note that how to use this mechanism has changed since Slurm 19.05.x (and
might change again later).

In this example we imagine that we wish to run a job with 5 MPI job steps
at the same time, each using 4 tasks, thus totalling to 20 tasks:

```{eval-rst}
.. literalinclude:: files/parallel_steps_cpu.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/parallel_steps_cpu.sh`
```

This will work with any {ref}`job-types` that hands out _cpus
and memory_, so that one specifies `--mem-per-cpu`.  For instance

    sbatch --partition=bigmem parallel_steps_cpu.sh

For job types that hand out _whole nodes_, notably the _normal_ jobs
on Fram and Betzy, one has to do it slightly different.  Here is an example to
run a `normal` job with 8 MPI job steps at the same time, each using
16 tasks, thus totalling 128 tasks:

```{eval-rst}
.. literalinclude:: files/parallel_steps_node.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/parallel_steps_node.sh`
```

For instance (on Fram):

    sbatch parallel_steps_node.sh

A couple of notes:

- The `wait` command is important - the run script will only continue once
  all commands started with `&` have completed.
- It is possible to use `mpirun` instead of `srun`, although `srun` is
  recommended for OpenMPI.
- The `export SLURM_MEM_PER_CPU=1920` and `unset SLURM_MEM_PER_NODE`
  lines prior to the `srun` lines are needed for jobs in the `normal` or
  `optimist` partitions on Fram and Betzy, because it is not possible
  to specify this to `sbatch` for such jobs.  Alternatively, you can
  add `--mem-per-cpu=1920` to the `srun` command lines (this only
  works with `srun`).  (1920 allows up to 32 tasks per node.  If each
  task needs more than 1920 MiB per cpu, the number must be increased
  (and the number of tasks per node will be reduced).  On *Betzy*, the
  corresponding number is 1960, which will allow up to 128 tasks per
  node.
- This technique does **not** work with IntelMPI, at least not when using
  `mpirun`, which is currently the recommended way of running IntelMPI jobs.
