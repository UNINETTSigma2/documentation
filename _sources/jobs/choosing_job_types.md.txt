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

| Name                                    | Description                               | Job limits   | Max walltime | Priority |
|:---------------------------------------:|-------------------------------------------|:------------:|:------------:|:--------:|
| {ref}`normal <job_type_betzy_normal>`   | default job type                          | 4--512 nodes | 4 days       | normal   |
| {ref}`accel <job_type_betzy_accel>`     | jobs needing GPUs                         |              | 7 days       | normal   |
| {ref}`preproc <job_type_betzy_preproc>` | pre-/postprocessing jobs                  | 1--16 units  | 1 day        | normal   |
| {ref}`devel <job_type_betzy_devel>`     | development jobs (compiling, testing)[^1] | 1--4 nodes   | 60 mins      | high     |

For jobs that don't request GPUs or much memory, the "units" of *accel* or *preproc* jobs
are simply the number of cpus the job requests.
For other jobs, see {ref}`projects-accounting` for how the units are calculated.

{ref}`Betzy Job Types <job-types-betzy>`.

## Olivia

| Name                                   | Description                               | Job limits                | Max walltime | Priority |
|:--------------------------------------:|-------------------------------------------|:-------------------------:|:------------:|:--------:|
| {ref}`normal <job_type_olivia_normal>` | default job type for CPU jobs             | 1--2304 units             | 7 days       | normal   |
| {ref}`accel <job_type_olivia_accel>`   | jobs needing GH200 GPUs (ARM64 nodes)     | 1--2304 units, 0--32 GPUs | 7 days       | normal   |
| {ref}`devel <job_type_olivia_devel>`    | development jobs (compiling, testing)[^2] | 1--1152 units, 0--16 GPUs| 2 hours      | high     |

"Units" are calculated based in the number of cpus and amount of memory
the job requests.  See {ref}`projects-accounting` for how.

{ref}`Olivia Job Types <job-types-olivia>`.

## Saga

| Name                                     | Description                               | Job limits   | Max walltime                                | Priority |
|:----------------------------------------:|-------------------------------------------|:------------:|:-------------------------------------------:|:--------:|
| {ref}`normal <job_type_saga_normal>`     | default job type                          | 1--256 units | 7 days                                      | normal   |
| {ref}`bigmem <job_type_saga_bigmem>`     | jobs needing more memory                  | 1--256 units | 14 days                                     | normal   |
| {ref}`hugemem <job_type_saga_hugemem>`   | jobs needing even more memory             | 1--256 units | 14 days                                     | normal   |
| {ref}`accel <job_type_saga_accel>`       | jobs needing P100 GPUs                    | 1--256 units | 14 days                                     | normal   |
| {ref}`a100 <job_type_saga_a100>`         | jobs needing A100 GPUs                    | 1--256 units | 14 days                                     | normal   |
| {ref}`devel <job_type_saga_devel>`       | development jobs (compiling, testing)[^2] | 1--128 units | 2 hours                                     | high     |
| {ref}`optimist <job_type_saga_optimist>` | jobs w/checkpointing, or very short jobs  | 1--256 units | {ref}`see details <job_type_saga_optimist>` | low      |

For jobs that don't request GPUs or much memory, the "units" on Saga are
simply the number of cpus the job requests.
For other jobs, see {ref}`projects-accounting` for how the units are calculated.

{ref}`Saga Job Types <job-types-saga>`.


[^1]: On Betzy it is possible to combine _devel_ with _accel_, {ref}`see details <job_type_betzy_devel>`.

[^2]: On Saga and Olivia it is possible to combine _devel_ with _accel_, _a100_,
    _bigmem_ or _hugemem_, {ref}`see details <job_type_saga_devel>`.
