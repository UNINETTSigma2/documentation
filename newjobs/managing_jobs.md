# Managing Jobs
- working with jobs (submit, inspect, cancel, requeue, managing (see
  fram doc))
  http://hpc.uit.no/en/latest/jobs/batch.html#managing-jobs
  http://hpc.uit.no/en/latest/jobs/monitoring.html

- link to Job Statuses

## Fra Abel
Inspecting Jobs

To get a quick view of the status of a job, you can use squeue:

squeue -j JobId

where JobId is the job id number that sbatch returns.  To see more details about a job, use

scontrol show job JobId

See man squeue and man scontrol for details about these commands.
