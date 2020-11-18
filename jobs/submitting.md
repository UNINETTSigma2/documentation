(submitting-jobs)=

# Submitting jobs

The HPC clusters are resources that are shared between many users, and
to ensure fair use everyone must do their computations by submitting
jobs through a queue system (batch system) that will execute the
applications on the available resources.
In our case [SLURM](https://slurm.schedmd.com/) is used as workload
manager and job scheduler.

When you log in to a cluster, you are logged in to a _login_ node
shared by all users. The login nodes are meant for logging in, copying
files, editing, compiling, running short tests (no more than a couple
of minutes), submitting jobs, checking job status, etc.
Jobs started via SLURM run on the _compute nodes_.

Note that it is _not_ allowed to run jobs directly on the login nodes.

## Jobs

There are two ways of starting jobs with SLURM; either interactively
with `srun` or as a job script with `sbatch`.

Interactive jobs are a good way to test your setup before you put it
into a script or to work with interactive applications like MATLAB or
python. You immediately see the results and can check if all parts
behave as you expected. See [interactive](interactive_jobs.md) for more
details.

To run a computations on the cluster, you submit a [job
script](job_scripts.md) into a _job queue_, and the job is started
when one or more suitable _compute nodes_ are available.


Job scripts are submitted with the
[sbatch](https://slurm.schedmd.com/sbatch.html) command:

    sbatch YourJobscript

The `sbatch` command returns a _jobid_, number that identifies the
submitted job. The job will be waiting in the job queue until there
are free compute resources it can use. A job in that state is said to
be _pending_ (PD). When it has started, it is called _running_ (R).
Any output (stdout or stderr) of the job script will be written to a
file called `slurm-<jobid>.out` in the directory where you ran
`sbatch`, unless otherwise specified.

It is also possible to pass arguments to the job script, like this:

    sbatch YourJobscript arg1 arg2

These will be available as the variables `$1`, `$2`, etc. in the job
script, so in this example, `$1` would have the value `arg1` and `$2`
the value `arg2`.

All commands in the job script are performed on the compute-node(s)
allocated by the queue system. The script also specifies a number of
requirements (memory usage, number of CPUs, run-time, etc.), used by
the queue system to find one or more suitable machines for the job.

### More information about SLURM
- For more information about the SLURM parameters and job script settings,
see [SLURM parameter](job_scripts/slurm_parameter.md).

- A more detailed description of the queue system can be found in
[Queue System Concepts](submitting/queue_system_concepts.md).

- If you are already used to PBS/Torque, but not Slurm, you might find
[Porting from PBS/Torque](guides/porting_from_pbs.md) useful.

## Job Queue

Jobs in the job queue are started on a priority basis, and a job gets
higher priority the longer it has to wait in the queue. A detailed
description can be found in [Job Scheduling](submitting/job_scheduling.md).

To see the list of running or pending jobs in the queue, use the
command [squeue](https://slurm.schedmd.com/squeue.html). Some useful `squeue` options:

    -j jobids   show only the specified jobs
    -w nodes    show only jobs on the specified nodes
    -A projects show only jobs belonging to the specified projects
    -t states   show only jobs in the specified states (pending, running,
                suspended, etc.)
    -u users    show only jobs belonging to the specified users

All specifications can be comma separated lists. Examples:

    squeue -j 14132,14133    # shows jobs 4132 and 4133
    squeue -w c23-11         # shows jobs running on c23-11
    squeue -u foo -t PD      # shows pending jobs belonging to user 'foo'
    squeue -A bar            # shows all jobs in the project 'bar'

To see all pending jobs, in priority order, you can use `pending`,
which is a small wrapper for `squeue`. See `pending --help` for
details and options.

For a description of common job states, see [Job States](monitoring/job_states.md).
