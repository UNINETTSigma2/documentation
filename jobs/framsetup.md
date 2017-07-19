# Fram Setup

The HPC machines use Slurm as its queue system. Slurm runs **jobs** from the
**queue**. The jobs contain steps and other set up information as defined
in a batch script file.

The [Job Scripts](jobscripts.md) page has more information about creating job scripts.

## The General Setup

Projects are administered and allocated resources based on the following:
1. **Partitions**: `normal`, `bigmem`, `accel` (later), `optimist`
2. **Account** "nn9999k" per project, with CPU hour quota. "nn9999x" are *optimist* jobs, with a **QoS** `optimist`
and separate CPU hour quotas.
3. **QoS** (Quality of Service). QoSes are administered to `normal`, `bigmem`
    and `accel` partitions. Special QoSes are `devel`, `preproc`.

## Account information

The project's account number is set in the batch script file as:

    #SBATCH --account=nn1234k

To print information on available CPU-hours in your accounts type in the terminal:

    $ cost

**Not implemented yet:** Total cpu usage per project: `qsumm`

## Job priority

The priority setup has been designed to be predictable and easy to
understand, and to try and maximize the utilisation of the cluster.

-   Principle: *Time-to-reservation*
-   Priority increases 1 point/minute
-   When priority is >= 20,000, job gets a reservation
-   **Devel** jobs start with 19,990, so reservation in 10 min
-   **Normal** jobs start with 19,940, so reservation in 60 min
-   **Normal unpri** jobs start with 19,400, so reservation in 10 hrs
-   **Optimist** jobs start with 1, ends at 10,080, so never reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other job should delay it. But before it gets a reservation,
the queue system backfiller is free to start any other job that can start now,
even if that will delay the job.

## Quality of Service (QoS)

A job's quality of service sets the partitions and different allocated resources to ensure efficient use of the machines.
For information about how to set the quality of service, visit the [Quality of Service](qos.md) page.

## Project Environment {#projenvironment}

-   Job gets private `/tmp` and `/var/tmp`, deleted by job epilog
-   Job prolog creates `$SCRATCH` (`/cluster/work/jobs/<jobid>`), deleted by epilog
-   Files marked with `copyback` *file* get copied back to submit directory by epilog
-   Commands registered with `cleanup` *command* get run by epilog
-   Prolog sets environment variables like `$SCRATCH`, `$OMP_NUM_THREADS`
-   Epilog prints usage statistics in the job stdout file

All jobs get a scratch area `$SCRATCH` that is created for the job and deleted
afterwards. To ensure that output files are copied back to the submit
directory, *even if the job crashes*, one should use the `copyback` command
*prior* to the main computation.

Alternatively, one can use the personal scratch area,
`/cluster/work/users/$USER`, where files will not be deleted after the job
finishes, but they *will be deleted* after 42 days. Note that
there is **no backup** on `/cluster/work/users/$USER` (or `$SCRATCH` directory).

For more information about storage options, click on the articles under the **Files and Storage** section on the menu to the left.
