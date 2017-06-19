Monitoring your jobs
====================

For details run the commands with the `--help` option.

List detailed information for a job (useful for troubleshooting):

    scontrol show jobid -dd <jobid>

To get statistics on pending or running jobs by jobID.

Once your job has completed, you can get additional information that was not
available during the run. This includes run time, memory used, etc:

    sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed

From our monitoring tool <?????>, you can watch live status information
on Fram:

-   Link to live monitoring

CPU load and memory consumption of your job
-------------------------------------------

In order to find out the CPU load and memory consumption of your running
jobs or jobs which have finished less than 48 hours ago, please use the
<???> LINK MISSING

Understanding your job status
-----------------------------

When you look at the job queue through <???> a link to live monitoring page, or
you use the `squeue` command, you will see that the queue is divided in
3 parts: Running jobs, Pending jobs or Held jobs.

Running jobs are the jobs that are running at the moment. Pending jobs are
next in line to start running, when the needed resources become available.
Held jobs are jobs that are put on hold by you, a staff member or the system.
The `Reason` field of `pending` or `scontrol show job` will indicate why a job
is held.

Please contact the support staff, if you don't understand why your job
has a hold state.
