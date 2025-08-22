(slurm_commands)=

# Useful Slurm commands and tools for managing jobs

## Slurm

Slurm, the job scheduler used on the HPC clusters, has a number of useful commands for managing jobs. Here is a growing collection of useful Slurm commands. Slurm commands can also be found in the official [Slurm documentation](https://slurm.schedmd.com/documentation.html).

### Commands for sbatch

```{note}
There are two ways of giving sbatch a command. One way is to include the command in the job script by adding `#SBATCH` before the command (just like you would do for the required sbatch commands such as `#SBATCH --job-name`, `#SBATCH --nodes`, etc.) The other way is to give the command in the command line when submitting a job script. For example for the command `--test-only`, you would submit the job with `sbatch --test-only job_script.sh`.
```

#### `--test-only`

Validates the script and report about any missing information (misspelled input files, invalid arguments, etc.) and give an estimate of when the job will start running. Will not actually submit the job to the queue.

#### `--gres=localscratch:<size>`

A job on **Fram or Saga** can request a scratch area on local disk on the node it is running on to speed up I/O intensive jobs. This command is not useful for jobs running on more than one node. Currently, there are no special commands to ensure that files are copied back automatically, so one has to do that with cp commands or similar in the job script. More information on using this command is found here: {ref}`job-scratch-area-on-local-disk`.

### Other Slurm commands

#### `sstat` and `sacct`

Job statistics can be found with `sstat` for _running_ jobs and with `sacct` for _completed_ jobs. In the command line, use `sstat` or `sacct` with the option `-j` followed by the job id number.

```console
$ sstat -j JobId
$ sacct -j JobId
```

Both `sstat` and `sacct` have an option `--format` to select which
fields to show. See the documentation on `sstat` [here](https://slurm.schedmd.com/sstat.html) and on `sacct` [here](https://slurm.schedmd.com/sacct.html).

## Tools

Here, a growing collection of useful tools available in the command line.

### seff

`seff` is a nice tool which we can use on **completed jobs**. For example here we ask
for a summary for the job number 4200691:

```console
$ seff 4200691
```

```{code-block}
---
emphasize-lines: 9-10
---
Job ID: 4200691
Cluster: saga
User/Group: user/user
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:01
CPU Efficiency: 2.70% of 00:00:37 core-walltime
Job Wall-clock time: 00:00:37
Memory Utilized: 3.06 GB
Memory Efficiency: 89.58% of 3.42 GB
```

### Slurm Job Script Generator

A tool for generating Slurm job scripts tailored for our HPC clusters: [Slurm Job Script Generator](https://open.pages.sigma2.no/job-script-generator/)
