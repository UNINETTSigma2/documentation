---
orphan: true
---

(job-types-saga)=

# Job Types on Saga

Saga is designed to run serial and small ("narrow") parallel jobs, in
addition to GPU jobs.  If you need to run "wider" parallel jobs, Fram
is a better choice.

```{warning}
**On Saga use srun, not mpirun**
     
mpirun can get the number of tasks wrong and also lead to wrong task    
placement. We don't fully understand why this happens. When using srun    
instead of mpirun or mpiexec, we observe correct task placement on Saga.      
```

The basic allocation units on Saga are cpu and memory.
The details about how the billing units are calculated can be found
in {ref}`projects-accounting`.

Most jobs on Saga are *normal* jobs.

Jobs requiring a lot of memory (> 8 GiB/cpu) should run as *bigmem*
or *hugemem* jobs.

Jobs that are very short, or implement checkpointing, can run as
*optimist* jobs, which means they can use resources that are idle for
a short time before they are requeued by a non-*optimist* job.

For development or testing, use a *devel* job

Here is a more detailed description of the different job types on
Saga:


(job_type_saga_normal)=

## Normal

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 256 units
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
    - 200 nodes with 40 cpus and 178.5 GiB RAM
	- 120 nodes with 52 cpus and 178.5 GiB RAM
- __Parameter for sbatch/salloc__:
    - None, _normal_ is the default
- __Job Scripts__: {ref}`job_scripts_saga_normal`

This is the default job type.  Most jobs are *normal* jobs.


(job_type_saga_bigmem)=

## Bigmem

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 256 units
- __Maximum walltime__: 14 days
- __Priority__: normal
- __Available resources__:
    - 28 nodes with 40 cpus and 362 GiB RAM
    - 8 nodes with 64 cpus and 3021 GiB RAM
- __Parameter for sbatch/salloc__:
    - `--partition=bigmem`
- __Job Scripts__: {ref}`job_scripts_saga_bigmem`

*Bigmem* jobs are meant for jobs that need a lot of memory (RAM),
typically more than 8 GiB per cpu.  (The _normal_ nodes on Saga have
slightly more than 4.5 GiB per cpu.)

Can be combined with `--qos=devel` to get higher priority but maximum wall time (2h) 
and resource limits of _devel_ apply.


(job_type_saga_hugemem)=

## Hugemem

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 256 units
- __Maximum walltime__: 14 days
- __Priority__: normal
- __Available resources__:
    - 2 nodes with 64 cpus and 6040 GiB RAM
- __Parameter for sbatch/salloc__:
    - `--partition=hugemem`
- __Job Scripts__: {ref}`job_scripts_saga_bigmem`

*Hugemem* jobs are meant for jobs that need even more memory (RAM)
than *bigmem* jobs.

Can be combined with `--qos=devel` to get higher priority but maximum wall time (2h) 
and resource limits of _devel_ apply.

Please note that not all of the ordinary software modules will work on
the *hugemem* nodes, due to the different cpu type.  If you encounter
any software-related issues, we are happy to help you at
support@nris.no.  As an alternative, you can use the[EESSI](https://www.eessi.io/docs/) module.  
These have been built to support the cpus on the hugemem nodes.  
To activate the module, do
`module load EESSI/2023.06` (EESSI)
before you load modules.


(job_type_saga_accel)=

## Accel

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
    - maximum 256 units
- __Maximum walltime__: 14 days
- __Priority__: normal
- __Available resources__: 8 nodes (max 7 per user) with 24 cpus, 364 GiB RAM and 4 P100
  GPUs.
- __Parameter for sbatch/salloc__:
    - `--partition=accel`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of GPUs
- __Job Scripts__: {ref}`job_scripts_saga_accel`

*Accel* jobs give access to use the P100 GPUs.

Can be combined with `--qos=devel` to get higher priority but maximum wall time (2h)
and resource limits of _devel_ apply.


(job_type_saga_a100)=

## A100

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
    - maximum 256 units
- __Maximum walltime__: 14 days
- __Priority__: normal
- __Available resources__: 8 nodes (max 7 per user) with 32 cpus, 1,000 GiB RAM and 4 A100
  GPUs.
- __Parameter for sbatch/salloc__:
    - `--partition=a100`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of GPUs
- __Job Scripts__: {ref}`job_scripts_saga_accel`

*A100* jobs give access to use the A100 GPUs.

Can be combined with `--qos=devel` to get higher priority but maximum wall time (2h)
and resource limits of _devel_ apply.


(job_type_saga_devel)=

## Devel

- __Allocation units__: cpus and memory and GPUs
- __Job Limits__:
    - maximum 128 units per job
    - maximum 256 units in use at the same time
    - maximum 2 running jobs per user
- __Maximum walltime__: 2 hours
- __Priority__: high
- __Available resources__: *devel* jobs can run on any node on Saga
- __Parameter for sbatch/salloc__:
    - `--qos=devel`
- __Job Scripts__: {ref}`job_scripts_saga_devel`

This is meant for small, short development or test jobs.  *Devel* jobs
get higher priority for them to run as soon as possible.  On the other
hand, there are limits on the size and number of _devel_ jobs.

Can be combined with either `--partition=accel`, `--partition=bigmem`
or `--partition=huemem` to increase
priority while having max wall time and job limits of _devel_ job.

If you have _temporary_ development needs that cannot be fulfilled by
the _devel_ or _short_ job types, please contact us at
[support@nris.no](mailto:support@nris.no).


(job_type_saga_optimist)=

## Optimist

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 256 units
- __Maximum Walltime__: None.  The jobs will start as soon as
  resources are available for at least 30 minutes, but can be
  requeued at any time, so there is no guaranteed minimum run time.
- __Priority__: low
- __Available resources__: *optimist* jobs can run on any node on Saga
- __Parameter for sbatch/salloc__:
    - `--qos=optimist`
- __Job Scripts__: {ref}`job_scripts_saga_optimist`

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
send a request to [support@nris.no](mailto:support@nris.no).
