# Job Types on Fram

Fram is designed to run medium-sized parallel jobs.  If you need to
run serial jobs or "narrow" parallel jobs, Saga is a better choice.

Most jobs on Fram are *normal* jobs.  For a preprocessing or
postprocessing job which only needs one node, use a *preproc* job.  If
it only needs a single cpu, a small *bigmem* job can be used instead.
Jobs requiring a lot of memory (FIXME: > 4 GiB/cpu? > 8 GiB/cpu?)
should run as *bigmem* jobs.  Jobs that are very short, or implement
checkpointing, can run as *optimist* jobs, which means they can use
resources that are idle for a short time before they are requeued by a
non-*optimist* job.  For development or testing, there are two job
types: *devel* usually has the shortest wait time during office hours
because it has access to dedicated resources, but it is limited to
small, short jobs.  *short* allows slightly larger and longer jobs,
but has no access to dedicated resources, so will probably have longer
wait times.

Here is a more detailed description of the different job types on
Fram:

- __normal__
    - __Description__: The default job type.  Most jobs are *normal* jobs.
    - __Allocation units__: whole nodes
    - __Job Limits__:
        - minimum 4 nodes, maximum 32 nodes (can be increased)
    - __Maximum walltime__: 7 days
    - __Priority__: normal
	- __Available resources__: 996 nodes with 32 cpus and 60 GiB RAM
- __preproc__
    - __Description__: Meant for small preprocessing or postprocessing
      jobs.  If the job is single-threaded, one can use a *bigmem* job
      instead, asking for 1 cpu.
	- __Allocation units__: whole nodes
	- __Job Limits__:
	    - min/max 1 node
	- __Maximum walltime__: 1 day
	- __Priority__: normal 
	- __Available resources__: *preproc* jobs run on the *normal* nodes
- __bigmem__
    - __Description__: Meant for jobs that need a lot of memory (RAM)
	- __Allocation units__: cpus and memory
    - __Job Limits__:
	    - (none)
	- __Maximum walltime__: 14 days
	- __Priority__: normal
	- __Available resources__:
	    - 8 nodes with 32 cpus and 501 GiB RAM
	    - 2 nodes with 64 cpus and 6045 GiB RAM
- <a name="optimist"></a>__optimist__
    - __Description__: Meant for jobs with checkpointing, or very
      short jobs.  *optimist* jobs will be started as soon as there
      are idle resources for at least 30 minutes, but will be requeued
      when other jobs need the resources (possibly before they have
      run 30 minutes).
    - __Allocation units__: whole nodes
	- __Job Limits__:
	    - minimum 4 nodes, maximum 32 nodes (can be increased)
	- __Maximum Walltime__: None.  The jobs will start as soon as
      resources are available for at least 30 minutes, but can be
      requeued at any time, so there is no guaranteed minimum run time.
    - __Priority__: low
	- __Available resources__: *optimist* jobs run on the *normal* nodes.
    - __Notes__: Projects need to make a request for using this job
      type, and will get a separate quota for it.
- __devel__
    - __Description__: meant for small, short development or test jobs.  Has
      access to a set of dedicated nodes on daytime in weekdays to
      make the jobs start as soon as possible.
    - __Allocation units__: whole nodes
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
    - __Allocation units__: whole nodes
	- __Job Limits__:
	    - minimum 1 nodes, maximum 10 nodes per job
	    - maximum 16 nodes in use at the same time
    - __Maximum walltime__: 2 hours
    - __Priority__: high (slightly lower than *devel*)
	- __Available resources__: 16 nodes with 32 cpus and 60 GiB RAM
      (shared with *normal*)
