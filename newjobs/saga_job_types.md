# Job Types on Saga

Saga is designed to run serial and small ("narrow") parallel jobs, in
addition to GPU jobs.  If you need to run "wider" parallel jobs, Fram
is a better choice.

The basic allocation units on Saga are cpu and memory.

Most jobs on Saga are *normal* jobs.  Jobs requiring a lot of memory
(FIXME: > 8 GiB/cpu?) should run as *bigmem* jobs.  Jobs that are very
short, or implement checkpointing, can run as *optimist* jobs, which
means they can use resources that are idle for a short time before
they are requeued by a non-*optimist* job.  For development or
testing, use a *devel* or *short* job.

Here is a more detailed description of the different job types on
Saga:

- __normal__
    - __Description__: The default job type.  Most jobs are *normal* jobs.
    - __Allocation units__: cpus and memory
    - __Job Limits__:
        - maximum 256 cpus
    - __Maximum walltime__: 7 days
    - __Priority__: normal
	- __Available resources__: 200 nodes with 40 cpus and 186 GiB RAM,
      in total 8000 cpus and 36.4 TiB RAM.
- __bigmem__
    - __Description__: Meant for jobs that need a lot of memory (RAM)
	- __Allocation units__: cpus and memory
    - __Job Limits__:
	    - (none)
	- __Maximum walltime__: 14 days
	- __Priority__: normal
	- __Available resources__:
	    - 28 nodes with 40 cpus and 377 GiB RAM
	    - 8 nodes with 64 cpus and 3021 GiB RAM
		- In total 1632 cpus and 33.9 TiB RAM.
- __accel__
    - __Description__: Meant for jobs needing GPUs
	- __Allocation units__: cpus, memory and GPUs
	- __Job Limits__:
	    - (none)
	- __Maximum walltime__: 14 days
	- __Priority__: normal
	- __Available resources__: 8 nodes with 24 cpus, 377 GiB RAM and 4
      GPUs, in total 192 cpus, 3019 GiB RAM and 32 GPUs.
- <a name="optimist"></a>__optimist__
    - __Description__: Meant for jobs with checkpointing, or very
      short jobs.  *optimist* jobs will be started as soon as there
      are idle resources for at least 30 minutes, but will be requeued
      when other jobs need the resources (possibly before they have
      run 30 minutes).
    - __Allocation units__: cpus and memory
	- __Job Limits__:
	    - maximum 256 cpus
	- __Maximum Walltime__: None.  The jobs will start as soon as
      resources are available for at least 30 minutes, but can be
      requeued at any time, so there is no guaranteed minimum run time.
    - __Priority__: low
	- __Available resources__: *optimist* jobs can run on any node on Saga
- __devel__
    - __Description__: meant for small, short development or test jobs.
    - __Allocation units__: cpus and memory and GPUs
    - __Job Limits__:
	    - maximum 64 cpus per job
	    - maximum 64 cpus in use at the same time
    - __Maximum walltime__: 30 minutes
    - __Priority__: high
	- __Available resources__: *devel* jobs can run on any node on Saga
- __short__
    - __Description__: meant for development or test jobs.  Allows
      slightly longer and wider jobs than *devel*, but has slightly
      lower priority.
    - __Allocation units__: cpus and memory and GPUs
	- __Job Limits__:
	    - maximum 128 cpus per job
	    - maximum 256 cpus in use at the same time
    - __Maximum walltime__: 2 hours
    - __Priority__: high (slightly lower than *devel*)
	- __Available resources__: *devel* jobs can run on any node on Saga
