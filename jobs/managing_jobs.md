# Managing Jobs

This page documents the tools one can use for working with jobs.

## Submitting Jobs

Jobs are submitted to the job queue with 
[sbatch](https://slurm.schedmd.com/sbatch.html), as described in the
[Queue System](queue_system.md) page.

## Inspecting Jobs

To check the job status of all your jobs, you can use
[squeue](https://slurm.schedmd.com/squeue.html):

    squeue -u MyUsername

You can also get a quick view of the status of a job

    squeue -j JobId

where `JobId` is the job id number that `sbatch` returns.  To see more
details about a job, use

    scontrol show job JobId

Both commands will show the job state (**ST**), and can show a job reason for
why a job is pending. [Job States](job_states.md) describes a few
of the more common ones.

While a job is running, it is possible to view some of its usage
statistics with the [sstat](https://slurm.schedmd.com/sstat.html)
command, and after it has finished,
[sacct](https://slurm.schedmd.com/sacct.html) will give you similar
information:

    sstat -j JobId
	sacct -j JobId

Both `sstat` and `sacct` have an option `--format` to select which
fields to show.  See the documentation of the commands for the
available fields and what they mean.

When a job has finished, the output file `slurm-JobId.out` will
contain some usage statistics from `sstat` and `sacct`.

On Fram, there is also the web tool [Job
Browser](https://desktop.fram.sigma2.no/slurmbrowser/html/squeue.html)
that can be used to inspect many aspects of the jobs.  (This will
later be added to Saga as well.)

## Inspecting Job Queue

Tools for inspecting the job queue are described in [Queue System](queue_system.md).

## Controlling jobs

You can cancel running or pending (waiting) jobs with [scancel](https://slurm.schedmd.com/scancel.html):

    scancel JobId                # Cancel job with id JobId (as returned from sbatch)
    scancel --user=MyUsername    # Cancel all your jobs
    scancel --account=MyProject  # Cancel all jobs in MyProject

The command [scontrol](https://slurm.schedmd.com/scontrol.html) can be
used to further control pending or running jobs:

- `scontrol requeue JobId`: Requeue a running job.  The job will be
  stopped, and its state changed to pending.
- `scontrol hold JobId`: Hold a pending job.  This prevents the queue
  system from starting the job.  The job reason will be set to `JobHeldUser`.
- `scontrol release JobId`: Release a held job.  This allows the queue
  system to start the job.

It is also possible to submit a job and put it on hold immediately
with `sbatch --hold JobScript`.
