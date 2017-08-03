# Managing Jobs

## Slurm Commands

This page lists the main commands for controlling and monitoring jobs:
For more details, run the commands with the `--help` option or visit the Slurm documentation at https://slurm.schedmd.com

## Monitoring Jobs

These commands give information about the status of jobs:

1. `scontrol show job <jobID>` - show information about a job
2. `squeue` - information about jobs in the queue system
3. `sacct` - statistics and accounting about a completed job in the accounting log file or Slurm database
4. `sinfo` - information about Slurm nodes and partitions. See [Node Information](qos.md#nodeinfo).
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

To see a list of pending or held jobs, use the `pending` command. The `REASON` field when using
`pending` or `scontrol show job` will indicate why a job is held.

Here are some common job REASON codes:

1. `Resources` - waiting for resources
2. `Priority` -  waiting for higher priority jobs to run
3. `JobHeldAdmin` - job has been put to hold (scontrol hold jobid) by an admin.
4. `ReqNodeNotAvail` - job has requested nodes that are (currently) not available.  It also lists all unavailable nodes (not only the ones the job has requested).

The squeue documentation has the list of Slurm job REASON codes: [REASON codes](https://slurm.schedmd.com/squeue.html#lbAF).

Please contact the support staff, if you don't understand why your job
has a hold state.

## Controlling Job Execution

The **scontrol** command is for viewing and controlling Slurm states. **scancel** is the command to cancel a job.

1. `scontrol requeue <jobid> ` - requeue a job
2. `scontrol hold <jobid> ` - hold job in pending
3. `scontrol release <jobid> ` - release a held job
4. `scancel <jobid>` - cancel a job

To see which state the job is in, use `scontrol show`, for example:

    scontrol show <jobid> -dd <jobid>

#### Holding and Releasing Jobs

A job on hold will not start or block other jobs from starting until you release the hold.
There are several ways of holding a job:

1. `sbatch --hold <script_name>` - submit a job script and put it on hold
2. `scontrol hold <job_id>` - hold a pending job

To release the job, use `scontrol release`:

    scontrol release <job_id>

## Splitting A Job into Tasks (Array Jobs)

***FIXME: This section needs rewrite:*** `arrayrun` is a local hack
implemented on Abel, from the time when Slurm didn't have array jobs. Now it
does, and so `arrayrun` has not been installed on Fram.  (The only reason it
has not been removed on Abel is for backward compatibility.) Use `sbatch
--array from-to jobscript.sm` (or `-a`).  This submits N copies of
`jobscript.sm`, each getting their own value of `$SLURM_ARRAY_TASK_ID`.  See
`man sbatch` for details.

To run many instances of the same job, use the __arrayrun__ command. This is useful if you have a lot of data-sets which you want to process in the same way with the same job-script. arrayrun works very similarly to mpirun:

    arrayrun [-r] from-to [sbatch switches] YourCommand

Typically, YourCommand is a compiled program or a job script. If it is a compiled program, the arrayrun command line must include the sbatch switches neccessary to submit the program. If it is a script, it can contain `#SBATCH` lines in addition to or instead of switches on the arrayrun command line. `from` and `to` are the first and last task number. Each instance of *YourCommand* can use the environment variable `$TASK_ID` for selecting which data set to use, etc. For instance:

    arrayrun 1-100 MyScript

will run 100 copies of *MyScript*, setting the environment variable `$TASK_ID` to 1, 2, ..., 100 in turn.
