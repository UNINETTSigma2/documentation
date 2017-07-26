# Job Scripts

The HPC machines run applications as jobs in the queuing system which schedules the tasks and process to run on compute nodes. Abel, Stallo, and Fram use the Slurm Workload Manager (Slurm). There is an example script found at [Sample Batch Script](samplescript.md).

For more information about the queue system and how to set the proper priorities, visit the [Queue System](queuesystem.md) page.

## Creating Job Scripts

A job script is a script file that contains commands to be executed by Slurm or
environment variables. The start of the file should specify the shell environment.

    #!/bin/sh

It can contain a mixture of shell commands, variables, and sbatch settings.

## Submitting Job Scripts

To submit a job, use the __sbatch__ command followed by the batch script.

    sbatch <scriptfile>

It is also possible to set arguments after the sbatch command. For more information
about sbatch, visit the *sbatch* documentation at: https://slurm.schedmd.com/sbatch.html

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

Here are a couple of examples for how to specify nodes and tasks for some
special cases:

1. 10 nodes, one task per cpu (typically single-threaded applications):

```
#SBATCH --nodes=10 --ntasks-per-node=32
```

2. 10 nodes, one task per node:

```
#SBATCH --nodes=10
```

3. 10 nodes, four tasks per node, one cpu per task (`$OMP_NUM_THREADS=1`):

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
| `SCRATCH`       | Scratch directory for each job. Read more at [Project Environment](framsetup.md#projenvironment)|
| `USERWORK`      | A permanent work area (`/cluster/work/users/$USER`).  It is advisable to create a subdirectory of USERWORK or use `$SCRATCH` instead. Read more at [Project Environment](framsetup.md#projenvironment)|
| `OMP_NUM_THREADS`       | The value specified in `--cpus-per-task`.Read more at [Quality of Service](qos.md).   |
