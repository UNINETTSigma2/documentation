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

When the job starts, it creates a file `slurm-<jobid>.out` where stdout and
stderr of the commands are logged (unless overridden by `sbatch` options).
While the job is running, you can check its progress with `cat slurm-<jobid>.out`.

Once your job has completed, you can get additional information that was not
available during the run. This includes run time, memory used, etc:

    sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed

### Interpreting the Slurm output file

Slurm will collect some CPU/memory/disk statistics regarding each job when it
completes. This information is printed to stdout, which means that it ends
up in the `slurm-<jobid>.out` file. The exact same information can be obtained
manually by the following Slurm commands:

    sacct -j <jobid> --format=Jobid,JobName,AllocCPUS,NTasks,MinCPU,MinCPUTask,AveCPU,Elapsed,ExitCode
    sacct -j <jobid> --format=Jobid,MaxRSS,MaxRSSTask,AveRSS,MaxPages,MaxPagesTask,AvePages
    sacct -j <jobid> --format=Jobid,MaxDiskRead,MaxDiskReadTask,AveDiskRead,MaxDiskWrite,MaxDiskWriteTask,AveDiskWrite

A lot more information is available by changing the `--format` keywords, see the
[slurm](https://slurm.schedmd.com/sacct.html#lbAF) documentation for further
details, under the "JOB ACCOUNTING FIELDS" section.

The printed output should be interpreted *by task*. E.g. the `MaxRSS` is the
maximum amount of memory used by any single task (not node, not core, but task)
during the execution, while `AveRSS` is the average over all tasks. If your
application runs in shared memory or hybrid parallel (e.g. MPI + OpenMP), note
that the shared memory threads do *not* count as individual tasks, as all
threads are sharing the resourses within each MPI task.

If you are debugging an "out of memory" issue, note that a single MPI task is
allowed to exceed its "share" of memory if there are more than one task per
node, as long as the *total* memory used by all tasks on a given node does not
exceed the allocated resources. This applies even if the memory was allocated
as `--mem-per-cpu`, since all resouces are anyway pooled together for all task
within a node.

### Job Status

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

### Holding and Releasing Jobs

A job on hold will not start or block other jobs from starting until you release the hold.
There are several ways of holding a job:

1. `sbatch --hold <script_name>` - submit a job script and put it on hold
2. `scontrol hold <job_id>` - hold a pending job

To release the job, use `scontrol release`:

    scontrol release <job_id>
