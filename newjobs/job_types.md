# Job Types

## Fram


| Name     | Description                    | Units        | Job limits  | Max walltime | Priority |
|----------|--------------------------------|--------------|-------------|--------------|----------|
| normal   | default job type               | node         | 4--32 nodes | 7 days       | normal   |
| preproc  | small pre-/postprocessing jobs | node         | 1 node      | 1 day        | normal   |
| bigmem   | jobs needing more memory       | cpu + memory | FIXME       | 14 days      | normal   |
| optimist | jobs w/checkpointing,          | node         | 4--32 nodes | see below    | low      |
|          | or very short jobs             |              |             |              |          |
| devel    | development jobs               | nodes        | 1--4 nodes  | 3 mins       | high     |
| short    | development jobs               | nodes        | 1--10 nodes | 2 hours      | high     |


- normal
    - default job type
    - allocation units: whole nodes
	- minimum 4 nodes, maximum 32 nodes (can be increased) per job
	- maximum walltime 7 days
    - normal priority
	- available resources: 996 nodes with 32 cpus and 60 GiB RAM
- preproc
    - for small preprocessing or postprocessing jobs
	- allocation units: whole nodes
	- max/min 1 node
	- maximum walltime 1 day
	- normal priority
	- available resources: 996 nodes with 32 cpus and 60 GiB RAM
      (shared with _normal_)
- bigmem
    - for jobs needing more memory
	- alloc units: cpus and memory
	- maximum walltime 14 days
	- normal priority
	- available resources:
	    - 8 nodes with 32 cpus and 501 GiB RAM
	    - 2 nodes with 64 cpus and 6045 GiB RAM
- optimist
    - for jobs with checkpointing, or very short jobs
    - allocation units: whole nodes
	- minimum 4 nodes, maximum 32 nodes (can be increased) per job
    - will run on idle resources, but be requeued when other jobs need
      the resources
	- will start as soon as resources are available for at least 30
      minutes, but no guaranteed run time before being requeued
    - low priority
	- available resources: 996 nodes with 32 cpus and 60 GiB RAM
      (shared with _normal_)
- devel
    - for development jobs
    - allocation units: whole nodes
    - has access to a set of reserved nodes during working hours to
      make the jobs start as soon as possible
	- minimum 1 nodes, maximum 4 nodes per job
	- maximum 4 nodes in use at the same time
    - maximum walltime 30 minutes
    - high priority
	- available resources: 4 nodes with 32 cpus and 60 GiB RAM between
      07:00 and 21:00 on weekdays
- short
    - for development jobs
    - allocation units: whole nodes
    - has access to a set of reserved nodes during working hours to
      make the jobs start as soon as possible
	- minimum 1 nodes, maximum 10 nodes per job
	- maximum 16 nodes in use at the same time
    - maximum walltime 2 hours
    - high priority
	- available resources: 16 nodes with 32 cpus and 60 GiB RAM
      (shared with _normal_)

## Saga

- normal
    - default job type
- bigmem
    - for jobs needing more memory
- accel
    - for jobs needing GPUs
- optimist
    - for jobs with checkpointing, or very short jobs
- devel
    - for development jobs




Describe the different types of jobs on each cluster, what they are supposed
to be used for, and which limits they have.  Perhaps in a separate page for
each cluster.  Use a standardized format.

http://hpc.uit.no/en/latest/jobs/partitions.html

## Fra Abel

General Job Limitations

Default values when nothing is specified:

    1 core (CPU)

The rest (time, mem per cpu, etc.) must be specified.

Limits

    Each project has its own limit of the number of concurrent used CPUs. See qsumm --group. (Notur projects share the same limit.)
    The max wall time is 1 week, except for jobs in the hugemem or long partitions, where it is 4 weeks. (definition of wall time)
    Max 400 submitted jobs per user at any time.
