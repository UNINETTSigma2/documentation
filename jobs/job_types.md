# Job Types

The clusters are designed for different work loads, and each cluster
has several types of jobs.  This page gives an overview of the
available job types on each cluster, and their main characteristics.
See the sub pages under the table for each cluster for more detailed
description of the job types and their purposes.

## Betzy

| Name                                            | Description                           | Job limits   | Max walltime                              | Priority |
|:-----------------------------------------------:|---------------------------------------|:------------:|:-----------------------------------------:|:--------:|
| [normal](job_types/betzy_job_types.md#normal)   | default job type                      | 8--256 nodes | 4 days                                    | normal   |
| [preproc](job_types/betzy_job_types.md#preproc) | pre-/postprocessing jobs              | 1 node       | 1 day                                     | normal   |
| [devel](job_types/betzy_job_types.md#devel)     | development jobs (compiling, testing) | 1--4 nodes   | 20 mins                                   | high     |

[Betzy Job Types](job_types/betzy_job_types.md).

## Fram

| Name                                             | Description                              | Job limits  | Max walltime                              | Priority |
|:------------------------------------------------:|------------------------------------------|:-----------:|:-----------------------------------------:|:--------:|
| [normal](job_types/fram_job_types.md#normal)     | default job type                         | 4--32 nodes | 7 days                                    | normal   |
| [preproc](job_types/fram_job_types.md#preproc)   | pre-/postprocessing jobs                 | 1 node      | 1 day                                     | normal   |
| [bigmem](job_types/fram_job_types.md#bigmem)     | jobs needing more memory                 |             | 14 days                                   | normal   |
| [devel](job_types/fram_job_types.md#devel)       | development jobs (compiling, testing)    | 1--4 nodes  | 30 mins                                   | high     |
| [short](job_types/fram_job_types.md#short)       | short jobs                               | 1--10 nodes | 2 hours                                   | high     |
| [optimist](job_types/fram_job_types.md#optimist) | jobs w/checkpointing, or very short jobs | 4--32 nodes | [see details](job_types/fram_job_types.md#optimist) | low      |

[Fram Job Types](job_types/fram_job_types.md).


## Saga

| Name                                             | Description                              | Job limits   | Max walltime                              | Priority |
|:------------------------------------------------:|------------------------------------------|:------------:|:-----------------------------------------:|:--------:|
| [normal](job_types/saga_job_types.md#normal)     | default job type                         | 1--256 units | 7 days                                    | normal   |
| [bigmem](job_types/saga_job_types.md#bigmem)     | jobs needing more memory                 | 1--256 units | 14 days                                   | normal   |
| [accel](job_types/saga_job_types.md#accel)       | jobs needing GPUs                        | 1--256 units | 14 days                                   | normal   |
| [devel](job_types/saga_job_types.md#devel)       | development jobs (compiling, testing)    | 1--128 units | 2 hours                                   | high     |
| [optimist](job_types/saga_job_types.md#optimist) | jobs w/checkpointing, or very short jobs | 1--256 units | [see details](job_types/saga_job_types.md#optimist) | low      |

For jobs that don't request GPUs or much memory, the units on Saga are
simply the number of cpus the job requests.  For other jobs, [see
here](projects_accounting.md#accounting) for how the units are calculated.

[Saga Job Types](job_types/saga_job_types.md).
