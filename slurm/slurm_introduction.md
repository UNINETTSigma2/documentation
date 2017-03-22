# Queue System

This page documents what is needed to get started with running jobs through the queue system Slurm on Fram.

In the following it is assumed that you

- know what a queue system is
- have used at least one (SGE, PBS/Torque, Slurm, ...)
- know how to read man pages

For basic Slurm concepts and Fram specific setup, see [here](slurm_concepts_and_setup_fram.md).


## Job Submission

-   Batch jobs: `sbatch` *jobscript*

-   Interactive jobs: `salloc` *specifications*

-   Easier-to-understand interactive jobs: `qlogin` *specifications* (our script)

-   Cancelling job: `scancel` *jobid*

-   Requeueing job: `scontrol requeue` *jobid*

-   Hold job in pending: `scontrol hold` *jobid*, `scontrol release` *jobid*

It is possible to ssh into compute nodes where you have a job running, but not to other nodes.

## General "normal" Job (1)

Must use between 4 and 30 nodes.  Gets whole nodes (all cpus and memory),
exclusively.

    #!/bin/bash
    ## Job parametres
    #SBATCH --account=nn9999k --time=1-0:0:0
    #SBATCH --nodes=10 --ntasks-per-node=2 --cpus-per-task=16
    
    ## Safety settings
    set -o errexit
    set -o nounset
    
    ## Software modules
    module purge   # Clear any inherited modules
    module load mysoftware

## General "normal" Job (2)

    ## Prepare input files
    cp inputfile $SCRATCH
    cd $SCRATCH
    
    ## Make sure output is copied back
    copyback outputfile
    
    ## Do some work
    srun mysoftware
    ## or
    mpirun mysoftware

## Simpler "normal" Jobs

10 nodes, one task per cpu:

    #SBATCH --account=nn9999k --time=1-0:0:0
    #SBATCH --nodes=10 --ntasks-per-node=32

10 nodes, one task per node:

    #SBATCH --account=nn9999k --time=1-0:0:0
    #SBATCH --nodes=10

## "bigmem" Job

Must specify cpus and memory.  Gets cpus and memory exclusively, but shares
nodes.

    #SBATCH --account=nn9999k --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --nodes=2 --ntasks-per-node=2 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

## "preproc" Jobs

Gets 1 node, exclusively.

General (1 node, 4 tasks with 8 cpus/task):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0
    #SBATCH --ntasks-per-node=4 --cpus-per-task=8

Simpler (one task on one node):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0

## Job and Queue Info

Jobs:

-   All job info: `scontrol show job` *jobid* (Add `-v` for more details, `-vv` for job script)

-   After job has finished: `sacct -j` *jobid* (se man page)

-   Updating job: `scontrol update jobid=<jobid>`

Queue:

-   Job queue: `squeue` (se man page)

-   Pending jobs: `pending` (our script)

-   Total cpu usage per project: `qsumm` (our script)

Accounting:

-   Project usage: `cost` (our script)

# Nodes

## Info about nodes

-   All nodes, all partitions: `sinfo`

-   Nodes in `normal` partition: `sinfo -p normal`

-   One node: `sinfo -n` *nodename*

-   See `sinfo` man page!

-   All info about node: `scontrol show node` *nodename*
