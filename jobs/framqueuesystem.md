# Queue System on Fram

The HPC machines use Slurm as its queue system. Slurm runs **jobs** from the
**queue**. The jobs contain steps and other set up information as defined
in a batch script file.

The [Job Scripts](jobscripts.md) page has more information about creating job scripts.

## The General Setup

Jobs are administered and allocated resources based on the following:
1. **Partitions** - logical groupings of nodes that can be thought of as queues.  `normal`, `bigmem`, `accel` (later), `optimist` partitions.
2. **QoS** (Quality of Service). QoSes are administered to `normal`, `bigmem`
    and `accel` partitions. Special QoSes are `devel`, `preproc`.
3. **Account** "nn9999k" per project, with CPU hour quota. "nn9999x" are *optimist* jobs, with a **QoS** `optimist`
and separate CPU hour quotas.


For an overview of the Slurm concepts, Slurm has an excellent beginners guide: [Quick Start](https://slurm.schedmd.com/quickstart.html).

## Job Types
Slurm accepts options to decide how to run applications. For example, setting the `--partition` option tells Slurm on which partition the job runs.
Mostly, the types correspond to the partition types, but other options give Slurm an idea about the resources applications need.

For information about setting the options, visit the [Job Types](jobtypes.md) page.


## Account information

The project's account number is set in the batch script file as:

    #SBATCH --account=nn1234k

To print information on available CPU-hours in your accounts type in the terminal:

    $ cost

A sample output might be:

    ============================================
    Account                            Cpu hours
    --------------------------------------------
    nnXXXXk  Quota (pri)                 1002.00
    nnXXXXk  Quota (nonpri)              1000.00
    nnXXXXk  Used                      112088.75
    nnXXXXk  Running                        0.00
    nnXXXXk  Pending                        0.00
    nnXXXXk  Available                -110086.75



**Not implemented yet:** Total cpu usage per project: `qsumm`

## Project Environment {#projenvironment}

Jobs have access to storage for storing temporary files and error output.

#### Work Directory

All jobs get a scratch area `$SCRATCH` that is created for the job and deleted
afterwards.

This is especially important if the job uses a lot of files, or does much random access on the files (repeatedly read and write to different places of the files).

There are several reasons for using $SCRATCH:

* $SCRATCH is on a faster file system than user home directories.
* There is less risk of interfering with running jobs by accidentally modifying or deleting the jobs' input or output files.
* Temporary files are automatically cleaned up, because the scratch directory is removed when the job finishes.
* It avoids taking unneeded backups of temporary and partial files, because $SCRATCH is not backed up.

The directory where you ran sbatch is stored in the environment variable `$SLURM_SUBMIT_DIR`.
The $SCRATCH directory is removed upon job exit (after copying back chkfiled files).

#### Prolog and Epilog

Slurm has several *Prolog* and *Epilog* programs that perform setup and cleanup
tasks, when a job or job step is run. On Fram, the Prolog and Epilog programs handle
the environment variables and cleanup of temporary files.

* Prolog sets environment variables like `$SCRATCH`, `$OMP_NUM_THREADS`
* Prolog creates `$SCRATCH` (`/cluster/work/jobs/<jobid>`), deleted later by Epilog
* Epilog deletes private `/tmp` and `/var/tmp`
* Epilog copies files marked with `copyback` *file* back to submit directory
* Epilog prints usage statistics in the job stdout file
* Epilog run commands registered with `cleanup <*command*>`

You can also use the command `cleanup` instead. It is used to specify commands to run when your job exits (before the $SCRATCH directory is removed).
For instance:

cleanup "cp $SCRATCH/outputfile /some/other/directory/newName"

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
