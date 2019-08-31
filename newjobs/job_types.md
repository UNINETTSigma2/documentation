# Job Types

The clusters are designed for different work loads, and each cluster
has several types of jobs.  This page gives an overview of the
available job types on each cluster, and their main characteristics.
See the sub pages under the table for each cluster for more detailed
description of the job types and their purposes.

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
| devel    | development jobs                                     | cpu + memory + GPU | 1--64 cpus  | 30 mins                                   | high     |
| short    | development jobs                                     | cpu + memory + GPU | 1--128 cpus | 2 hours                                   | high     |

FIXME: Do we really need the *short* job type on Saga?  Are the limits
for *devel* ok?

[Saga Job Types](saga_job_types.md).
