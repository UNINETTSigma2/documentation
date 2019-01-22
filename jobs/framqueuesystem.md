# Queue System on Fram

The HPC machines use Slurm as its queue system. Slurm runs **jobs** from the
**queue**. The jobs contain steps and other set up information as defined
in a batch script file.

The [Job Scripts](jobscripts.md) page has more information about creating job scripts.

## The General Setup

Jobs are administered and allocated resources based on the following:
1. **Partitions** - logical groupings of nodes that can be thought of as queues.  `normal`, `bigmem`, `optimist` partitions.
2. **QoS** (Quality of Service). QoSes are administered to `normal` and `bigmem`
	partitions. Special QoSes are `devel`, `preproc`.
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

## Project Environment

Jobs have access to temporary storage, backedup storage, environment variables, and
software modules (read [Installed Software](../development/which_software_is_installed.md)) for use during execution.

#### Work Directory

All jobs can use a work area (`$SCRATCH` and `$USERWORK` variables) that is created for the job and deleted
afterwards. The work area is mounted on a fast file system that can handle large amounts of input and output.

The directory where you ran sbatch is stored in the environment variable `$SLURM_SUBMIT_DIR`. This variable is suitable for
copying files relative to the current directory.

For more information about storing files on Fram, visit [Storage Systems on Fram](../storage/storagesystems.md).

#### Prolog and Epilog {#prolog_epilog}

Slurm has several *Prolog* and *Epilog* programs that perform setup and cleanup
tasks, when a job or job step is run. On Fram, the Prolog and Epilog programs handle
the environment variables and cleanup of temporary files.

* Prolog sets environment variables like `$SCRATCH`, `$OMP_NUM_THREADS`
* Prolog creates `$SCRATCH` (`/cluster/work/jobs/<jobid>`), deleted later by Epilog
* Epilog deletes private `/tmp` and `/var/tmp`
* Epilog copies files marked with `savefile` *file* back to submit directory
* Epilog prints usage statistics in the job stdout file
* Epilog run commands registered with `cleanup <*command*>`

You can also use the command `cleanup` instead. It is used to specify commands to run when your job exits (before the $SCRATCH directory is removed).
For instance:

cleanup "cp $SCRATCH/outputfile /some/other/directory/newName"

## Job priority

The priority setup has been designed to be predictable and easy to
understand, and to try and maximize the utilisation of the cluster.

-   Principle: *Time-to-reservation*
-   Priority increases 1 point/minute[^1]
-   When priority is >= 20,000, job gets a reservation
-   **Devel** jobs start with 20,000, so reservation directly
-   **Normal** jobs start with 19,760, so reservation in 4 hours
-   **Normal unpri** jobs start with 19,040, so reservation in 16 hrs
-   **Optimist** jobs start with 1, ends at 10,080, so never reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other job should delay it. But before it gets a reservation,
the queue system backfiller is free to start any other job that can start now,
even if that will delay the job.

[^1]: Currently, only the priority of 10 jobs for each user
    within each project increase with time.  As jobs start, more
    priorities start to increase.  This is done in order to avoid
    problems if a user submits a large amount of jobs over a short
    time.  Note that the limit is per user and project, so if a user
    has jobs in several projects, 10 of the user's jobs from each
    project will increase in priority at the same time.  This limit
    might change in the future.

## Job Placement

The compute nodes on Fram are divided into four groups, or *islands*.  The
network bandwidth within an island is higher than the throughput between
islands.  Some jobs need high network throughput between its nodes, and will
usually run faster if they run within a single island.  Therefore, the queue
system is configured to run each job within one island, if possible.  See
[Job Placement on Fram](framjobplacement.md) for details and for how this can
be overridden.
