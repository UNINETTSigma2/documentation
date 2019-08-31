# Job Types on Saga

Saga is designed to run serial and small ("narrow") parallel jobs, in
addition to GPU jobs.  If you need to run "wider" parallel jobs, Fram
is a better choice.

The basic allocation units on Saga are cpu and memory.

Most jobs on Saga are *normal* jobs.  Jobs requiring a lot of memory
(FIXME: how much?) should run as *bigmem* jobs.  Jobs that are very
short, or implement checkpointing, can run as *optimist* jobs, which
means they can use resources that are idle for a short time before
they are requeued by a non-*optimist* job.  For development or
testing, use a *devel* job.

Here is a more detailed description of the different job types on
Saga:

- __normal__
    - __Description__: The default job type.  Most jobs are *normal* jobs.
    - __Allocation units__: cpus and memory
    - __Job Limits__:
        - maximum 256 cpus
    - __Maximum walltime__: 7 days
    - __Priority__: normal
	- __Available resources__: FIXME 996 nodes with 32 cpus and 60 GiB RAM
- __bigmem__
    - __Description__: Meant for jobs that need a lot of memory (RAM)
	- __Allocation units__: cpus and memory
    - __Job Limits__:
	    - FIXME
	- __Maximum walltime__: 14 days
	- __Priority__: normal
	- __Available resources__:
	    - 8 nodes with 32 cpus and 501 GiB RAM
	    - 2 nodes with 64 cpus and 6045 GiB RAM
- __preproc__
    - __Description__: Meant for small preprocessing or postprocessing
      jobs.  If the job is single-threaded, one can use a *bigmem* job
      instead, asking for 1 cpu.
	- __Allocation units__: cpus and memory
	- __Job Limits__:
	    - min/max 1 node
	- __Maximum walltime__: 1 day
	- __Priority__: normal 
	- __Available resources__: 996 nodes with 32 cpus and 60 GiB RAM
      (shared with *normal*)
- __optimist__
    - __Description__: Meant for jobs with checkpointing, or very
      short jobs.  *optimist* jobs will be started as soon as there
      are idle resources for at least 30 minutes, but will be requeued
      when other jobs need the resources (possibly before they have
      run 30 minutes).
    - __Allocation units__: cpus and memory
	- __Job Limits__:
	    - minimum 4 nodes, maximum 32 nodes (can be increased)
	- __Maximum Walltime__: None.  The jobs will start as soon as
      resources are available for at least 30 minutes, but can be
      requeued at any time, so there is no guaranteed minimum run time.
    - __Priority__: low
	- __Available resources__: 996 nodes with 32 cpus and 60 GiB RAM
      (shared with *normal*)
- __devel__
    - __Description__: meant for small, short development or test jobs.  Has
      access to a set of dedicated nodes on daytime in weekdays to
      make the jobs start as soon as possible.
    - __Allocation units__: cpus and memory  FIXME: Can one use bigmem?
    - __Job Limits__:
	    - minimum 1 nodes, maximum 4 nodes per job
	    - maximum 4 nodes in use at the same time
    - __Maximum walltime__: 30 minutes
    - __Priority__: high
	- __Available resources__: 4 nodes with 32 cpus and 60 GiB RAM between
      07:00 and 21:00 on weekdays
- __short__
    - __Description__: meant for development or test jobs.  Allows
      slightly longer and wider jobs than *devel*, but has slightly
      lower priority, and no dedicated resources.
    - __Allocation units__: cpus and memory  FIXME: Can one use bigmem?
	- __Job Limits__:
	    - minimum 1 nodes, maximum 10 nodes per job
	    - maximum 16 nodes in use at the same time
    - __Maximum walltime__: 2 hours
    - __Priority__: high (slightly lower than *devel*)
	- __Available resources__: 16 nodes with 32 cpus and 60 GiB RAM
      (shared with *normal*)

