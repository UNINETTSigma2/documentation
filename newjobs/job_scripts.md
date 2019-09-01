# Job Scripts

This page documents the basics of how to write job scripts for the HPC clusters.
Cluster-specific details are kept in separate sub pages for each cluster.

## Job Script Basics

To run a _job_ on the cluster involves creating a shell script called
a _job script_.  The job script is a plain-text file containing any
number of commands, including your main computational task, i.e., it
may copy or rename files, cd into the proper directory, etc., all
before doing the "real" work.  The lines in the script file are the
commands to be executed, in the given order.  Lines starting with a
`#` are ignored as comments, except lines that start with `#SBATCH`,
which are not executed, but contain special instructions to the [queue
system](queue_system.md).

If you are not familiar with shell scripts, they are simply a set of
commands that you could have typed at the command line.  You can find
more information about shell scripts here: [Introduction to Bash shell
scripts](http://www.linuxconfig.org/Bash_scripting_Tutorial).

A job script consists of a couple of parts, in this order:

- The first line, which is always `#!/bin/bash`[^1]
- Paremeters to the queue system
- Commands to set up the execution environment
- The actual commands you want to be run

Parameters to the queue system may be specified on the `sbatch`
command line and/or in `#SBATCH` lines in the job script.  There can
be as many `#SBATCH` lines as you want, and you can combine several
parameters on the same line.  If a parameter is specified both on the
command line and in the job script, the parameter specified on the
command line takes precedence.  The `#SBATCH` lines must precede any
commands in the script.  Which parameters are allowed or compulsory
depends the job type.  One parameter must be present in any job:
`--account`, which specifies the *project* the job will run in.  The
other parameters will be described in the sub pages for each cluster.

It is recommended to start the commands to set up the environment with

```bash
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module purge    # Reset the modules to the system default (FIXME: what is the lmod way of saying this?)
```

and will most likely include one or more

```bash
module load <em>SomeProgram/SomeVersion</em>
```

to set up environment variables like `$PATH` to get access to the
specified programs.  It is recommended to specify the explicit version
in the `module load` command.  We also recommend adding a

```bash
module list    # For easier debugging
```

after the `module load` commands.

All in all, a generic job script might look like this:

[include](files/generic_job_script.sh)

Download the script: <a
href="files/generic_job_script.sh">generic_job_script.sh</a> (you might have
to right-click and select `Save Link As...` or similar).

## Footnotes

[^1] Technically, any script language that uses `#` as a comment
character can be used, but the recommended, and the only one supported
by the Metacenter, is bash.

## Notes

http://hpc.uit.no/en/latest/jobs/batch.html

- Basic, common info
  Describe available env. variables
  http://hpc.uit.no/en/latest/jobs/environment-variables.html

- work directory (scratch, userwork, project area, copyback, cleanup,
  alternative with signal before timout)

- wall time?
  http://hpc.uit.no/en/latest/jobs/batch.html#walltime

- job dependencies

- array jobs
  http://hpc.uit.no/en/latest/jobs/examples.html

- parallel tasks (perhaps per cluster)
  http://hpc.uit.no/en/latest/jobs/examples.html

- mpi jobs (perhaps per cluster)
  http://hpc.uit.no/en/latest/jobs/running_mpi_jobs.html

- checkpointing

- link to porting from pbs/torque tutorial

- A sub page for each cluster, showing all details and examples for each job
  type.
  For Fram: job placement.
  http://hpc.uit.no/en/latest/jobs/examples.html

## Questions

- Some examples use `#!/bin/bash -l`.  Why? and Should we recommend/use
  that?
- Should we recommend/use an explicit `exit 0` at the end?
- I have to right-click and select `Save Link As...` to download
  script.  Is that just me?
