# Queue System

This page documents what is needed to get started with running jobs through the Slurm queue system on Fram.

In the following it is assumed that you

- know what a queue system is
- have used at least one (SGE, PBS/Torque, Slurm, ...)

For basic Slurm concepts and Fram specific setup, see [here](slurm_concepts_and_setup_fram.md).


## Job Submission

Job scripts is submitted to the queue system with `sbatch` *jobscript*.

It is possible to run interactive jobs with `salloc` *specifications*, where
*specifications* are the same specifications that would normally be used in a
job script.  Note that commands run inside a `salloc` job are executed on the
*login node* where you ran `salloc`.  To get processes executed on the
allocated compute nodes, one must use `srun` or `mpirun`, for instance `srun
hostname`.

**This has not been implemented yet:** There is also a command `qlogin` for
running interactive jobs where you get a shell on the main allocated compute
node: `qlogin` *specifications*.  This can be less confusing than `salloc`.
Note, however, that behind the scenes, `qlogin` uses `srun` to start the
shell.  That means that when using `srun` or `mpirun` one might have to
specify the number of tasks manually.  So for testing with `srun` or `mpirun`,
it is recommended to use `salloc`.

## Manipulating Jobs

- Cancelling job: `scancel` *jobid*
- Requeueing job: `scontrol requeue` *jobid*
- Hold job in pending: `scontrol hold` *jobid*
- Release a held job: `scontrol release` *jobid*

**This has not been implemented yet:** Note that it is possible to ssh into
compute nodes where you have a job running, but not to other nodes.

## Job Scripts

There are different types of jobs:

- *normal*: Most production jobs, runs on all standard compute nodes.
- *bigmem*: Jobs on the bigmem nodes.
- **Not implemented yet:** *accel*: Jobs om compute nodes with GPU cards.
- *optimist*: Low priority jobs that can be requeued when higher priority jobs
  need their nodes.  They must implement checkpointing.

### General "normal" Jobs

Normal jobs must specify account, walltime limit and number of nodes.  By
default, jobs must use between 4 and 30 nodes, but a project can be allowed to
use more nodes if needed.  It can also specify how many tasks should run per
node, and/or how many cpus should be used by each task.  Other things like job
name can also be specified.

All normal jobs gets exclusive access to whole nodes (all cpus and memory).
If a job tries to use more (resident) memory than is configured on the nodes,
it will be killed. **FIXME: what is the limit?** The maximal wall time limit
for normal jobs is two days (48 hours).

Here is an example job script which specifies "everyting":

    #!/bin/bash
    ## Project:
    #SBATCH --account=nnNNNNk
	## Job name:
	#SBATCH --jobname=myjob
	## Wall time limit:
	#SBATCH --time=1-0:0:0
	## Number of nodes:
    #SBATCH --nodes=10
	## Number of tasks to start on each node:
	#SBATCH --ntasks-per-node=2
	## Set OMP_NUM_THREADS
	#SBATCH --cpus-per-task=16
    
    ## Recommended safety settings:
    set -o errexit # Make bash exit on any error
    set -o nounset # Treat unset variables as errors
    
    ## Software modules
    module purge   # Clear any inherited modules
    module load mysoftware

    ## Prepare input files
    cp inputfiles $SCRATCH
    cd $SCRATCH
    
    ## Make sure output is copied back after job finishes
    copyback outputfile1 outputfile2
    
    ## Do some work
    srun mysoftware
    ## or
    mpirun mysoftware

Note that `--cpus-per-task` does *not* bind the tasks to the given number of
cpus for normal jobs; it merely sets `$OMP_NUM_THREADS` so that OpenMP jobs by
default will use the right number of threads.  It is possible to override this
number by setting `$OMP_NUM_THREADS` in the job script.

All jobs get a scratch area `$SCRATCH` that is created for the job and deleted
afterwards.  To ensure that output files are copied back to the submit
directory, *even if the job crashes*, one should use the `copyback` command
*prior* to the main comutation.

Alternatively, one can use the personal scratch area,
`/cluster/work/users/$USER`, where files will not be deleted after the job
finishes, but they *will be deleted* after **FIXME** days.

See `man sbatch` for more details about job specifications.

### Simpler "normal" Job Specifications

Here are a couple of examples for how to specify nodes and tasks for some
special cases:

10 nodes, one task per cpu (typically single-threaded applications):

    #SBATCH --nodes=10 --ntasks-per-node=32

10 nodes, one task per node:

    #SBATCH --nodes=10

10 nodes, four tasks per node, one cpu per task (`$OMP_NUM_THREADS=1`):

    #SBATCH --nodes=10 --ntasks-per-node=4

### "bigmem" Jobs

To use the bigmem nodes (**FIXME: how many and how big**),
`--partition=bigmem`, cpus and memory must be specified.  A bigmem job gets
the requested cpus and memory exclusively, but shares
nodes with other jobs.

    #SBATCH --account=nn9999k --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --nodes=2 --ntasks-per-node=2 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

### "preproc" Jobs

There is a special QoS set up for preprocessing jobs (`--qos=preproc`). A preproc job gets 1 node, exclusively.

General (1 node, 4 tasks with 8 cpus/task):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0
    #SBATCH --ntasks-per-node=4 --cpus-per-task=8

Simpler (one task on one node):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0

## Job and Queue Info

Commands to get more information about jobs:

-   All job info: `scontrol show job` *jobid* (Add `-v` for more details, `-vv` for job script)

-   After job has finished: `sacct -j` *jobid*

-   Updating job: `scontrol update jobid=<jobid>`

Commands to get more information about the job queue:

-   Job queue: `squeue`

-   Pending jobs: `pending`

-   Total cpu usage per project: `qsumm`

To get and overview of how much your project(s) have used:

-   Project usage: `cost`

## Info about nodes

-   All nodes, all partitions: `sinfo`

-   Nodes in `normal` partition: `sinfo -p normal`

-   One node: `sinfo -n` *nodename*

-   All info about node: `scontrol show node` *nodename*
