(job-types)=

# Job Types

The clusters are designed for different work loads, and each cluster
has several types of jobs.  This page gives an overview of the
available job types on each cluster, and their main characteristics.
See the sub pages of each cluster for a more detailed description of
the job types and their purposes.

You should always choose the job type that fulfils the resource
requirements of your job best without being excessive as this ensures
the highest possible priority and therefore the shortest queuing time.
If for example your job on Saga needs 20GB memory and 5 CPUs for 5 days,
you should choose a _normal_ instead of a _bigmem_ job.
If it instead needs 200GB memory you should use _bigmem_.

## Betzy

| Name                                     | Description                           | Job limits   | Max walltime | Priority |
|:----------------------------------------:|---------------------------------------|:------------:|:------------:|:--------:|
| {ref}`normal <job_type_betzy_normal>`    | default job type                      | 4--512 nodes | 4 days       | normal   |
| {ref}`preproc <job_type_betzy_preproc>`  | pre-/postprocessing jobs              | 1--8 units   | 1 day        | normal   |
| {ref}`devel <job_type_betzy_devel>`      | development jobs (compiling, testing) | 1--4 nodes   | 60 mins      | high     |

For jobs that don't request much memory, the "units" of *preproc* jobs
are simply the number of cpus the job requests.
For other jobs, see {ref}`job_accounting` for how the units are calculated.

[Betzy Job Types](job_types/betzy_job_types.md).

## Fram

| Name                                     | Description                              | Job limits  | Max walltime                              | Priority |
|:----------------------------------------:|------------------------------------------|:-----------:|:-------------------------------------------:|:--------:|
| {ref}`normal <job_type_fram_normal>`     | default job type                         | 1--32 nodes | 7 days                                      | normal   |
| {ref}`bigmem <job_type_fram_bigmem>`     | jobs needing more memory                 |             | 14 days                                     | normal   |
| {ref}`devel <job_type_fram_devel>`       | development jobs (compiling, testing)    | 1--8 nodes  | 30 mins                                     | high     |
| {ref}`short <job_type_fram_short>`       | short jobs                               | 1--10 nodes | 2 hours                                     | high     |
| {ref}`optimist <job_type_fram_optimist>` | jobs w/checkpointing, or very short jobs | 1--32 nodes | {ref}`see details <job_type_fram_optimist>` | low      |

[Fram Job Types](job_types/fram_job_types.md).


## Saga

| Name                                             | Description                               | Job limits   | Max walltime                              | Priority |
|:------------------------------------------------:|-------------------------------------------|:------------:|:-----------------------------------------:|:--------:|
| {ref}`normal <job_type_saga_normal>`     | default job type                          | 1--256 units | 7 days                                      | normal   |
| {ref}`bigmem <job_type_saga_bigmem>`     | jobs needing more memory                  | 1--256 units | 14 days                                     | normal   |
| {ref}`accel <job_type_saga_accel>`       | jobs needing GPUs                         | 1--256 units | 14 days                                     | normal   |
| {ref}`devel <job_type_saga_devel>`       | development jobs (compiling, testing)[^1] | 1--128 units | 2 hours                                     | high     |
| {ref}`optimist <job_type_saga_optimist>` | jobs w/checkpointing, or very short jobs  | 1--256 units | {ref}`see details <job_type_saga_optimist>` | low      |

For jobs that don't request GPUs or much memory, the "units" on Saga are
simply the number of cpus the job requests.
For other jobs, see {ref}`job_accounting` for how the units are calculated.

[Saga Job Types](job_types/saga_job_types.md).

[^1]: On Saga it is possible to combine _devel_ with _accel_ or _bigmem_, {ref}`see details <job_type_saga_devel>`.
