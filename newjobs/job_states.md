# Job States

Commands like `squeue`, `sacct` and `scontrol show job` will show a
_state_ of each job.  All job states are explained in the [JOB STATE
CODES section of the squeue documentation
page](https://slurm.schedmd.com/squeue.html#lbAG).

Here is a table with the most common ones

| Name | Long name     | Description                                                         |
|:----:|:-------------:|---------------------------------------------------------------------|
| PD   | Pending       | Job is waiting to be started                                        |
| CF   | Configuring   | The queue system is starting up the job                             |
| R    | Running       | Job is running                                                      |
| CG   | Completing    | Job is finishing                                                    |
| CD   | Completed     | Job has finished                                                    |
| CA   | Cancelled     | Job has been cancelled, either before or after it started           |
| F    | Failed        | Job has exited with a non-zero exit status                          |
| TO   | Timeout       | Job didn't finish in time, and was cancelled                        |
| PR   | Preemepted    | Job was requeued because a higher priority job needed the resources |
| NF   | Node_fail     | Job was requeued because of a problem with one of its comput nodes  |
| OOM  | Out_of_memory | Job was cancelled because it tried to use too much memory           |

The commands can also give a _reason_ why a job is in the state it is.
This is most useful for pending jobs.  All these reasons are explained
in the [JOB REASON CODES section of the squeue documentation
page](https://slurm.schedmd.com/squeue.html#lbAF).

Here is a table with the most common ones

| Name                    | Description                                                                                                          |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|
| Resources               | The job is waiting for resources to become idle                                                                      |
| Priority                | There are jobs with higher priority than this job.  The job might be started, if it does not delay any of those jobs |
| AssocGrpCPUMinutesLimit | (On Fram) There is not enough hours left on the quota to start the job                                               |
| AssocBillingMinutes     | (On Saga) There is not enough hours left on the quota to start the job                                               |
| ReqNodeNotAvail         | One or more of the job's required nodes is currently not available, typically because it is down or reserved         |
| Dependency              | The job is waiting for jobs it depend on to start or finish.                                                         |
| JobHeldUser             | The job has been put on hold by the user                                                                             |
| JobHeldAdmin            | The job has been put on hold by an admin.  Please contact support if you don't know why it is being held.            |
