# Job Types

The clusters are designed for different work loads, and each cluster
has several types of jobs.  This page gives an overview of the
available job types on each cluster, and their main characteristics.
See the sub pages under the table for each cluster for more detailed
description of the job types and their purposes.

## Betzy

| Name     | Description                                          | Units        | Job limits   | Max walltime                              | Priority |
|:--------:|------------------------------------------------------|:------------:|:------------:|:-----------------------------------------:|:--------:|
| normal   | default job type                                     | node         | 8--256 nodes | 4 days                                    | normal   |
| preproc  | pre-/postprocessing jobs                             | node         | 1 node       | 1 day                                     | normal   |
| devel    | development jobs                                     | nodes        | 1--4 nodes   | 20 mins                                   | high     |

[Betzy Job Types](job_types/betzy_job_types.md).

## Fram

| Name     | Description                                          | Units        | Job limits  | Max walltime                              | Priority |
|:--------:|------------------------------------------------------|:------------:|:-----------:|:-----------------------------------------:|:--------:|
| normal   | default job type                                     | node         | 4--32 nodes | 7 days                                    | normal   |
| preproc  | pre-/postprocessing jobs                             | node         | 1 node      | 1 day                                     | normal   |
| bigmem   | jobs needing more memory                             | cpu + memory |             | 14 days                                   | normal   |
| optimist | <p>jobs w/checkpointing,<br/> or very short jobs</p> | node         | 4--32 nodes | [see details](job_types/fram_job_types.md#optimist) | low      |
| devel    | development jobs                                     | nodes        | 1--4 nodes  | 30 mins                                   | high     |
| short    | development jobs                                     | nodes        | 1--10 nodes | 2 hours                                   | high     |

[Fram Job Types](job_types/fram_job_types.md).


## Saga

| Name     | Description                                          | Units              | Job limits  | Max walltime                              | Priority |
|:--------:|------------------------------------------------------|:------------------:|:-----------:|:-----------------------------------------:|:--------:|
| normal   | default job type                                     | cpu + memory       | 1--256 units | 7 days                                    | normal   |
| bigmem   | jobs needing more memory                             | cpu + memory       | 1--256 units | 14 days                                   | normal   |
| accel    | jobs needing GPUs                                    | cpu + memory + GPU | 1--256 units | 14 days                                   | normal   |
| optimist | <p>jobs w/checkpointing,<br/> or very short jobs</p> | cpu + memory       | 1--256 units | [see details](job_types/saga_job_types.md#optimist) | low      |
| devel    | development jobs                                     | cpu + memory + GPU | 1--128 units | 2 hours                                   | high     |

For jobs that don't request GPUs or much memory, the units on Saga are
simply the number of cpus the job requests.  For other jobs, [see
here](projects_accounting.md#accounting) for how the units are calculated.

[Saga Job Types](job_types/saga_job_types.md).
