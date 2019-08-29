# Queue System

When you log in to the cluster, you are logged in to a _login_ node
shared by all users. The login nodes are meant for logging in, copying
files, editing, compiling, running short tests (no more than a couple
of minutes), submitting jobs, checking job status, etc.

To run a job on the cluster, you submit a [job script](job_scripts.md)
into a _job queue_, and the job is started when one or more suitable
_compute nodes_ are available.  The job queue is managed by a queue
system (scheduler and resource manager) called
[Slurm](https://slurm.schedmd.com/).

Note that it is _not_ allowed to run jobs directly on the login nodes.

Job scripts are submitted with the [sbatch](https://slurm.schedmd.com/sbatch.html) command:

    sbatch YourJobscript

The `sbatch` command returns a _jobid_, number that identifies the
submitted job.  The job will be waiting in the job queue until there
are free compute resources it can use.  A job in that state is said to
be _pending_ (PD).  When it has started, it is called _running_ (R).
Any output (stdout or stderr) of the job script will be written to a
file called `slurm-<jobid>.out` in the directory where you ran
`sbatch`, unless otherwise specified.

All commands in the job script are performed on the compute-node(s)
allocated by the queue system.  The script also specifies a number of
requirements (memory usage, number of CPUs, run-time, etc.), used by
the queue system to find one or more suitable machines for the job.

You can cancel running or pending (waiting) jobs with [scancel](https://slurm.schedmd.com/scancel.html):

    scancel JobId                # Cancel job with id JobId (as returned from sbatch)
    scancel --user=MyUsername    # Cancel all your jobs
    scancel --account=MyProject  # Cancel all jobs in MyProject

For more information about managing jobs, see [Managing Jobs](managing_jobs.md).
