# Job Scripts

The HPC machines run applications as jobs in the queuing system which schedules the tasks and process to run on compute nodes. Machines in the Metacenter do currently use the [Slurm Workload Manager](https://slurm.schedmd.com/). There is an example script found at [Sample MPI Batch Script](sample_mpi_script.md).

For more information about the queue system and how to set the proper priorities, visit the [Queue System](queuesystem.md) page.

## Creating Job Scripts

A job script is a script file that contains commands to be executed by Slurm or
environment variables. The start of the file should specify the shell environment.

    #!/bin/bash

It can contain a mixture of shell commands, variables, and sbatch settings.

## Submitting Job Scripts

To submit a job, use the __sbatch__ command followed by the batch script.

    sbatch <scriptfile>

The sbatch command returns a **jobid**, an id number that identifies the submitted job. The job will be waiting in the job queue until there are free compute resources it can use. A job in that state is said to be pending (PD). When it has started, it is called running (R). Any output (stdout or stderr) of the job script will be written to a file called *slurm-jobid.out* in the directory where you ran sbatch, unless otherwise specified.

All commands in the job script are performed on the compute-node(s) allocated by the queue system. The script also specifies a number of requirements (memory usage, number of CPUs, run-time, etc.), used by the queue system to find one or more suitable machines for your job.

To see the status of the submitted job, use the command `pending` and for related commands, read the [Managing Jobs](managing_jobs.md)

For more information about *sbatch*, visit the sbatch documentation at: https://slurm.schedmd.com/sbatch.html

## sbatch variables

**sbatch** variables begin with `#SBATCH` and commands are entered verbatim as
running in a shell environment. sbatch variables set up information about the job
such as account information and priority levels.

All batch scripts must contain the following sbatch variables:

    #Project name as seen in the queue
    #SBATCH --account=<project_name>

    #Application running time. The wall clock time helps the scheduler assess priorities and tasks.
    #SBATCH --time=<wall_clock_time>

    #Memory usage per core
    #SBATCH --mem-per-cpu=<size_megabytes>

* `--account` - specifies the project the job will run in.
* `--time` - specifies the maximal wall clock time of the job. Do not specify the limit too low, because the job will be killed if it has not finished within this time. On the other hand, shorter jobs will be started sooner, so do not specify longer than you need. The default maximum allowed --time specification is 1 week; see details here.
* `--mem-per-cpu` - specifies how much RAM each task (default is 1 task) of the job needs.

Here are a couple of examples for how to specify nodes and tasks for some
special cases:

* 10 nodes, one task per cpu (typically single-threaded applications):

```
#SBATCH --nodes=10 --ntasks-per-node=32
```

* 10 nodes, one task per node:

```
#SBATCH --nodes=10
```

* 10 nodes, four tasks per node, one cpu per task (`$OMP_NUM_THREADS=1`):

```
#SBATCH --nodes=10 --ntasks-per-node=4
```

For a sample job script, visit [Sample Batch Script](samplescript.md).

## Environment Variables

Slurm sets environment variables when running the batch script. These variables provide additional
parameters and information to be used by applications within the script.

| Variable | Description     |
| :------------- | :------------- |
| `SLURM_JOB_ID`      | Job ID of running job  |
| `SLURM_JOB_NODELIST` | List of nodes used in a job |

To use an environment variable, prepend the variable with the __$__ as you would in shell scripts. For example, to
print the job ID, enter the following in the batch script:

    echo $SLURM_JOB_ID

For a list of Slurm's environment variables, visit the sbatch documentation at: https://slurm.schedmd.com/sbatch.html

In addition, these are additional environment variables set in the HPC machines.

| Variable | Description     |
| :------------- | :------------- |
| `SCRATCH`       | Scratch directory for each job. Read more at [Work Area](../storage/storagesystems.md)|
| `USERWORK`      | Temporary work area (`/cluster/work/users/$USER`). Used for sharing data across jobs. Read more at [Work Area](../storage/storagesystems.md#workarea)
| `OMP_NUM_THREADS`       | The value specified in `--cpus-per-task`.Read more at [Quality of Service](qos.md).   |
