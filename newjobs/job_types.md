# Job Types

The clusters are designed for different work loads, and each cluster
defines different types of jobs.  This page gives an overview of the
available job types on each cluster, and their main characteristics.
See the sub pages for each cluster for more detailed description of
the job types and their purposes.

## Fram


| Name     | Description                                          | Units        | Job limits  | Max walltime                              | Priority |
|:--------:|------------------------------------------------------|:------------:|:-----------:|:-----------------------------------------:|:--------:|
| normal   | default job type                                     | node         | 4--32 nodes | 7 days                                    | normal   |
| preproc  | pre-/postprocessing jobs                             | node         | 1 node      | 1 day                                     | normal   |
| bigmem   | jobs needing more memory                             | cpu + memory |             | 14 days                                   | normal   |
| optimist | <p>jobs w/checkpointing,<br/> or very short jobs</p> | node         | 4--32 nodes | [see details](fram_job_types.md#optimist) | low      |
| devel    | development jobs                                     | nodes        | 1--4 nodes  | 30 mins                                   | high     |
| short    | development jobs                                     | nodes        | 1--10 nodes | 2 hours                                   | high     |

[Fram Job Types](fram_job_types.md).

## Saga

| Name     | Description                                          | Units              | Job limits  | Max walltime                              | Priority |
|:--------:|------------------------------------------------------|:------------------:|:-----------:|:-----------------------------------------:|:--------:|
| normal   | default job type                                     | cpu + memory       | 1--256 cpus | 7 days                                    | normal   |
| bigmem   | jobs needing more memory                             | cpu + memory       | (none)      | 14 days                                   | normal   |
| accel    | jobs needing GPUs                                    | cpu + memory + GPU | (none)      | 14 days                                   | normal   |
| optimist | <p>jobs w/checkpointing,<br/> or very short jobs</p> | cpu + memory       | 1-256 cpus  | [see details](saga_job_types.md#optimist) | low      |
| devel    | development jobs                                     | nodes              | FIXME       | 30 mins                                   | high     |

[Saga Job Types](fram_job_types.md).


# Notes:

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
