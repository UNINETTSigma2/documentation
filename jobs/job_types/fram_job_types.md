(job-types-fram)=

# Job Types on Fram

Fram is designed to run medium-sized parallel jobs.  If you need to
run serial jobs or "narrow" parallel jobs, Saga is a better choice.

Most jobs on Fram are *normal* jobs.

Jobs requiring a lot of memory (> 4 GiB/cpu) should run as *bigmem*
jobs. Also, jobs requiring only a single cpu, can use a small *bigmem* job.

Jobs that are very short, or implement checkpointing, can run as
*optimist* jobs, which means they can use resources that are idle for
a short time before they are requeued by a non-*optimist* job.

For development or testing, there are two job types: *devel* usually
has the shortest wait time during office hours, but is limited to
small, short jobs.  *short* allows slightly larger and longer jobs,
but will probably have longer wait times.

Here is a more detailed description of the different job types on
Fram:


(job_type_fram_normal)=

## Normal

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 node, maximum 32 nodes (can be increased)
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 996 nodes with 32 cpus and 60 GiB RAM
- __Parameter for sbatch/srun__:
    - None, _normal_ is the default
- __Job Scripts__: {ref}`job_scripts_fram_normal`

This is the default job type.  Most jobs are *normal* jobs.  Most of
the other job types are "variants" of a *normal* job.

In _normal_ jobs, the queue system hands out complete nodes.  If a
project needs more than 32 nodes per job, and the application in
question can actually scale more than 32 nodes, please send a request
to [support@metacenter.no](mailto:support@metacenter.no).


(job_type_fram_bigmem)=

## Bigmem

- __Allocation units__: cpus and memory
- __Job Limits__:
    - (none)
- __Maximum walltime__: 14 days
- __Priority__: normal
- __Available resources__:
    - 8 nodes with 32 cpus and 501 GiB RAM
    - 2 nodes with 64 cpus and 6045 GiB RAM
- __Parameter for sbatch/srun__:
    - `--partition=bigmem`
- __Job Scripts__: {ref}`job_scripts_fram_bigmem`

*Bigmem* jobs are meant for jobs that need a lot of memory (RAM),
typically more than 4 GiB per cpu.  (The _normal_ nodes on Fram have
slightly less than 2 GiB per cpu.)

For _bigmem_ jobs, the queue system hands out cpus and memory, not
whole nodes.


(job_type_fram_devel)=

## Devel

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 nodes, maximum 8 nodes per job
    - maximum 8 nodes in use at the same time
- __Maximum walltime__: 30 minutes
- __Priority__: high
- __Available resources__: 8 nodes with 32 cpus and 60 GiB RAM between
  07:00 and 21:00 on weekdays
- __Parameter for sbatch/srun__: 
    - `--qos=devel`
- __Job Scripts__: {ref}`job_scripts_fram_devel`

This is meant for small, short development or test jobs.  *Devel* jobs
have access to a set of dedicated nodes on daytime in weekdays to
make the jobs start as soon as possible.  On the other hand, there are
limits on the size and number of _devel_ jobs.

If you have _temporary_ development needs that cannot be fulfilled by
the _devel_ or _short_ job types, please contact us at
[support@metacenter.no](mailto:support@metacenter.no).


(job_type_fram_short)=

## Short

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 nodes, maximum 10 nodes per job
    - maximum 16 nodes in use at the same time
- __Maximum walltime__: 2 hours
- __Priority__: high (slightly lower than *devel*)
- __Available resources__: 16 nodes with 32 cpus and 60 GiB RAM
  (shared with *normal*)
- __Parameter for sbatch/srun__: 
    - `--qos=short`
- __Job Scripts__: {ref}`job_scripts_fram_short`

This is also meant for development or test jobs.  It allows slightly
longer and wider jobs than *devel*, but has slightly lower priority,
and no dedicated resources.  This usually results in a longer wait
time than *devel* jobs, at least on work days.


(job_type_fram_optimist)=

## Optimist

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 node, maximum 32 nodes (can be increased)
- __Maximum Walltime__: None.  The jobs will start as soon as
  resources are available for at least 30 minutes, but can be
  requeued at any time, so there is no guaranteed minimum run time.
- __Priority__: low
- __Available resources__: *optimist* jobs run on the *normal* nodes.
- __Parameter for sbatch/srun__: 
    - `--partition=optimist`
- __Job Scripts__: {ref}`job_scripts_fram_optimist`

The _optimist_ job type is meant for very short jobs, or jobs with
checkpointing (i.e., they save state regularly, so they can restart
from where they left off).

_Optimist_ jobs get lower priority than other jobs, but will start as
soon as there are free resources for at least 30 minutes.  However,
when any other non-_optimist_ job needs its resources, the _optimist_
job is stopped and put back on the job queue.  This can happen before
the _optimist_ job has run 30 minutes, so there is no _guaranteed_
minimum run time.

Therefore, all _optimist_ jobs must use checkpointing, and access to
run _optimist_ jobs will only be given to projects that demonstrate
that they can use checkpointing.  If you want to run _optimist_ jobs,
send a request to [support@metacenter.no](mailto:support@metacenter.no).
