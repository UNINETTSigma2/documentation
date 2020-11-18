

# Common job failures

Although users run very different types of jobes on the clusters, two errors are common and here we
describe how to detect these as well as steps to fix these common job failures.


## Running out of memory

A job will stop if it tries to use more memory than requested from Slurm.

This error is reported in the Slurm output:
```
slurm_script: line 11: 33333 Killed ./mybinary
slurmstepd: error: Detected 1 oom-kill event(s) in step 997857.batch cgroup.
Some of your processes may have been killed by the cgroup out-of-memory
handler.
```

The cryptic but interesting information here is `oom-kill` ("oom" is short for "out of memory") and `out-of-memory`.

You can fix this by requesting more memory in you job script:
```
#SBATCH --mem-per-cpu=1G
```

**But don't ask for way too much memory either**, otherwise you can get billed
for a lot more than you use, and your jobs may queue for a lot longer than you
would like to. In addition this can also block resources for others.

[Here](choosing_memory_settings.md) you can find out
how much memory your job really needs.


## Disk quota exceeded

Since the clusters are shared resources with hundreds of users, we have to have
quotas in place to prevent any user or group from
using too much disk space and making the system unusable for others.

When a group or user reaches their disk quota, files cannot be created due to the cause `Disk Quota Exceeded`.
This will often stop jobs that need to write output or log files.

There could be different quota settings for your home folder, project folders,
and other folders in addition to a file count or size quota.  Please consult
[this page](/files_storage/clusters.md) for
more details to see how to inspect your disk quota.

(Here we need to link to policies on how/when to ask for more)


## Job is rejected because of insufficient funds

If you see an error message like this one after submitting your job script:
```
sbatch: error: AssocGrpBillingMinutes
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy
               (job submit limit, user's size and/or time limits)
```

Then check with `cost` whether your compute account has enough funds for your
job.  The error probably means that you as for more resources in your job
script than you have available.
