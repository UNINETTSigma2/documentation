(job-scripts)=

# Job Scripts

This page documents the basics of how to write job scripts for the HPC clusters.
Cluster-specific details are kept in separate sub pages for each cluster:

- [Fram job scripts](job-scripts-on-fram)
- [Saga job scripts](job-scripts-on-saga)

```{note}
Email notification from completed Slurm scripts is currently disabled **on all
machines** and it looks like it will take quite a while (months?) before we can
re-enable it. Sorry for the inconvenience. The reason is technical due to the
way the infrastructure is set up. It is non-trivial for us to re-enable this in
a good and robust and secure way.
```


## Job Script Basics

To run a _job_ on the cluster involves creating a shell script called
a _job script_.  The job script is a plain-text file containing any
number of commands, including your main computational task, i.e., it
may copy or rename files, cd into the proper directory, etc., all
before doing the "real" work.  The lines in the script file are the
commands to be executed, in the given order.  Lines starting with a
`#` are ignored as comments, except lines that start with `#SBATCH`,
which are not executed, but contain special instructions to the queue
system.

If you are not familiar with shell scripts, they are simply a set of
commands that you could have typed at the command line.  You can find
more information about shell scripts here: [Introduction to Bash shell
scripts](http://www.linuxconfig.org/Bash_scripting_Tutorial).

A job script consists of a couple of parts, in this order:

- The first line, which is typically `#!/bin/bash`
  (the Slurm script does not have to be written in Bash, see below)
- Parameters to the queue system
- Commands to set up the execution environment
- The actual commands you want to be run

Parameters to the queue system may be specified on the `sbatch`
command line and/or in `#SBATCH` lines in the job script.  There can
be as many `#SBATCH` lines as you want, and you can combine several
parameters on the same line.  If a parameter is specified both on the
command line and in the job script, the parameter specified on the
command line takes precedence.  The `#SBATCH` lines must precede any
commands in the script.

Which parameters are allowed or required depends the job type and
cluster, but two parameters must be present in (almost) any job:

- `--account`, which specifies the *project* the job will run in.
  Required by all jobs.
- `--time`, which specifies how long a job should be allowed to
  run.  If it has not finished within that time, it will be cancelled.

The other parameters will be described in the sub pages for each cluster.

It is recommended to start the commands to set up the environment with

```bash
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
```

and will most likely include one or more

```bash
module load SomeProgram/SomeVersion
```

to set up environment variables like `$PATH` to get access to the
specified programs.  It is recommended to specify the explicit version
in the `module load` command.  We also recommend adding a

```bash
module list    # For easier debugging
```

after the `module load` commands. See also {ref}`module-scheme`.

All in all, a generic job script might look like this:

```{eval-rst}
.. literalinclude:: files/generic_job.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/generic_job.sh`
```

## Wall Time Limit
The wall time limit (`--time`) is required for all jobs.

The most used formats for the time specification is `DD-HH:MM:SS`
and `HH:MM:SS`, where *DD* is days, *HH* hours, *MM* minutes and *SS*
seconds.  For instance:

- `3-12:00:00`: 3.5 days
- `7:30:00`: 7.5 hours

We recommend you to be as precise as you can when specifying the wall
time limit as it will inflict on how fast your jobs will start to
run:  It is easier for a short job to get started between two larger,
higher priority jobs (so-called *backfilling*).  On the other hand, if
the job has not finished before the wall time limit, it will be
cancelled, so too long is better than too short due to lost work!


## The Slurm script does not have to be written in Bash

The job script can be written in
any language that uses `#` as the comment sign. Bash is most common, but some
applications like NorESM use Python. Perl, Julia, and R are other options. Here
is a Python example:
```python
#!/usr/bin/env python

#SBATCH --job-name=slurm
#SBATCH --account=nn9999k
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:0:5

import os

os.system("srun hostname")
```

Using Python, Perl, Julia, or R can open up for more programming possibilities
within the run script than what would be possible using Bash.


## Further Topics

- [Environment variables available in job scripts](job_scripts/environment_variables.md)
- [Job work directory](job_scripts/work_directory.md)
- [Array jobs](job_scripts/array_jobs.md)
- [Running Job Steps in Parallel](guides/running_job_steps_parallel.md)
- [Porting Job Scripts from PBS/Torque](guides/porting_from_pbs.md)
- [Running MPI Jobs](guides/running_mpi_jobs.md)
