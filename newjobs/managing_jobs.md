# Managing Jobs

## Submitting Jobs
sbatch

## Inspecting Jobs
Inspecting Jobs

To get a quick view of the status of a job, you can use squeue:

squeue -j JobId

where JobId is the job id number that sbatch returns.  To see more details about a job, use

scontrol show job JobId

See man squeue and man scontrol for details about these commands.


squeue, scontrol show job, sstat(?), sacct(?), mention job browser,

- link to Job Statuses

### Finished Jobs
sacct(?)

sacct-information in the end of job script,

## Inspecting Job Queue
FIXME: Perhaps just link to the queue system page!

squeue, mention job browser, pending

## Controlling jobs
scancel, requeue, hold/release

You can cancel running or pending (waiting) jobs with [scancel](https://slurm.schedmd.com/scancel.html):

    scancel JobId                # Cancel job with id JobId (as returned from sbatch)
    scancel --user=MyUsername    # Cancel all your jobs
    scancel --account=MyProject  # Cancel all jobs in MyProject


## Notes

- see https://documentation.sigma2.no/jobs/managing_jobs.html

