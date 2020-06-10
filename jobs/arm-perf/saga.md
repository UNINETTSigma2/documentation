

# Using Arm Performance Reports on Saga

<div class="alert alert-danger">
  <p>
    Arm Performance Reports on Saga currently only works in combination with the
    <b>Intel toolchain</b> for jobs that run <b>on one node</b>.

    We are working on restoring also other combinations and profiling jobs
    that run on more than one node. We will update this
    documentation once we get it to work correctly.
  </p>
</div>

<div class="alert alert-danger">
  <p>
    Arm Performance Reports does not work in the "express launch mode".  In
    other words <b>it cannot be used in combination with commands starting with
    "mpirun" or "srun"</b>.
    Only use it in the "compatibility mode" by giving the number of MPI tasks
    as an argument to perf-report.
  </p>
</div>

Use Arm Performance Reports only either in job scripts or on an interactive
compute node **never on a login node**:

- [Profiling a batch script](#profiling-a-batch-script)
- [Profiling on an interactive compute node](#profiling-on-an-interactive-compute-node)


### Profiling a batch script

Let us consider the following example job script as your
usual computation which you wish to profile:

```bash
#!/bin/bash -l

#SBATCH --account=YourAccount
#SBATCH --job-name=without-apr
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=4 --ntasks-per-node=4
#SBATCH --qos=devel

# recommended bash safety settings
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

srun ./myexample.x  # <- we will need to modify this line
```

All we need to do is to load the `Arm-PerfReports/20.0.3` module
and to modify the `srun` command to instead use
[perf-report](https://developer.arm.com/docs/101137/latest/running-with-an-example-program)
(you need to adjust "YourAccount"):

```bash
#!/bin/bash -l

#SBATCH --account=YourAccount
#SBATCH --job-name=with-apr
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=4 --ntasks-per-node=4
#SBATCH --qos=devel

# recommended bash safety settings
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module load Arm-PerfReports/20.0.3  # <- we added this line

perf-report -n ${SLURM_NTASKS} ./myexample.x  # <- we modified this line
```

In other words replace `srun` or `mpirun -n ${SLURM_NTASKS}` by
`perf-report -n ${SLURM_NTASKS}`.

That's it.


### Profiling on an interactive compute node

To run interactive tests one needs to submit
[an interactive job](/jobs/interactive_jobs.md)
to Slurm using `srun` (**not** using `salloc`), e.g.:

```bash
# obtain an interactive compute node for 30 minutes
# adjust "YourAccount"
$ srun --ntasks=1 --mem-per-cpu=1G --time=00:30:00 --qos=devel --account=YourAccount --pty bash -i

# load the module
$ module load Arm-PerfReports/20.0.3

# profile my application
$ perf-report -n ${SLURM_NTASKS} ./myexample.x
```
