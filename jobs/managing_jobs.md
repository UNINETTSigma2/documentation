# Managing Jobs

## Slurm Commands

This page lists the main commands for controlling and monitoring jobs:
For more details, run the commands with the `--help` option or visit the Slurm documentation at https://slurm.schedmd.com

## Monitoring Jobs

These commands give information about the status of jobs:

1. `scontrol show job <jobID>` - show information about a job
2. `squeue` - information about jobs in the queue system
3. `sacct` - statistics and accounting about a completed job in the accounting log file or Slurm database
4. `sinfo` - information about Slurm nodes and partitions. See [Info about nodes](../slurm/slurm_introduction.md#nodeinfo).
5. `pending` - list of pending jobs

For details run the commands with the `--help` option or visit the Slurm documentation at https://slurm.schedmd.com

Once your job has completed, you can get additional information that was not
available during the run. This includes run time, memory used, etc:

    sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed

#### Job Status

The queue is divided in 3 parts: **Running** jobs, **Pending** jobs or **Held** jobs.

Running jobs are the jobs that are running at the moment. Pending jobs are
next in line to start running, when the needed resources become available.
Held jobs are jobs that are put on hold by you, a staff member or the system.

To see a list of pending or held jobs, use the `pending command.` The `REASON` field when using
`pending` or `scontrol show job` will indicate why a job is held.

Please contact the support staff, if you don't understand why your job
has a hold state.

## Controlling Job Execution

The **scontrol** is for viewing and controlling Slurm states. **scancel** is the command to cancel a job.

1. Requeue a job: `scontrol requeue <jobid> `
2. Hold job in pending: `scontrol hold <jobid> `
3. Release a held job: `scontrol release <jobid> `
4. Cancel a job: `scancel <jobid>`

To see which state the job is in, use `scontrol show`, for example:

    scontrol show <jobid> -dd <jobid>

#### Holding and Releasing Jobs

A job on hold will not start or block other jobs from starting until you release the hold.
There are several ways of holding a job:

1. `sbatch --hold <script_name>` - submit a job script and put it on hold
2. `scontrol hold <job_id>` - hold a pending job

To release the job, use `scontrol release`:

	scontrol release <job_id>
