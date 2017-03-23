# Slurm Concepts

## Basic Slurm Concepts

-   **Partition**: Set of nodes (perhaps overlapping), job limits
-   **QoS** (Quality of Service): Priority, job limits
-   **Account**: A "project" (*nn9999k*), used for accounting, can have access to QoSes
-   **User**: Unix user, can have access to accounts
-   **Job**: Owned by a user, runs in an account, runs with a QoS in a partition
-   There is only *one* job queue(!)

## Jobs

-   Job script (`sbatch`) or interactive (`salloc` or `qlogin` **not
    implemented yet**)
-   Submitted to the job queue
-   A job is divided into job **steps**, run sequentially, started by `srun` or
      `mpirun` (think: "do this, then do that")
-   Each job step is divided into **tasks**, run in parallel (e.g., MPI ranks)

# Setup on Fram

## The General Setup

-   **Partitions**: `normal`, `bigmem`, `accel` (later), `optimist`
-   One **account** "nn9999k" per project, with cpu hour quota
-   One **QoS** per project, with default priority.  Has access to `normal`, `bigmem`
    and `accel` partitions
-   Special **QoSes** `devel`, `preproc`, for development, preprocessing, etc.
-   Special **accounts** "nn9999x" for *optimist* jobs, with a **QoS** `optimist`
    and separate cpu hour quotas

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
reservation, no other job should delay it.  But before it gets a reservation,
the queue system backfiller is free to start any other job that can start now,
even if that will delay the job.

## Job setup

-   Job gets private `/tmp` and `/var/tmp`, deleted by job epilog
-   Job prolog creates `$SCRATCH` (`/cluster/work/jobs/<jobid>`), deleted by epilog
-   Files marked with `copyback` *file* get copied back to submit directory by epilog
-   Commands registered with `cleanup` *command* get run by epilog
-   Prolog sets environment variables like `$SCRATCH`, `$OMP_NUM_THREADS`
-   Epilog prints usage statistics in the job stdout file
