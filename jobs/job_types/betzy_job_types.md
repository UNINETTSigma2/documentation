---
orphan: true
---

(job-types-betzy)=

# Job Types on Betzy

Betzy is designed to run highly parallelized jobs.  If you need to run medium-sized jobs, than Fram is a better choice, while for serial jobs you shall use Saga.

For a preprocessing or postprocessing job which only needs one or a few CPU cores, use a *preproc* job.

For development or testing use the  *devel* queue which is limited to small and short jobs. 

Here is a more detailed description of the different job types on Betzy:


(job_type_betzy_normal)=

## Normal

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 4 nodes
    - maximum 512 nodes
- __Maximum walltime__: 4 days
- __Priority__: normal
- __Available resources__: 1340 nodes, each with 128 CPU cores and 244 GiB RAM
- __Parameter for sbatch/salloc__:
    - None, _normal_ is the default
- __Job Scripts__: {ref}`job_scripts_betzy_normal`

This is the default job type. In _normal_ jobs, the queue system hands out complete nodes.

(job_type_betzy_accel)=

## Accel
- __Allocation units__: CPUs, Memory and GPUs
- __Job Limits__:
- __Maximum walltime__: 7 days
- __Priority__: Normal
- __Available resources__: 4 nodes, each with 64 CPU cores, 494.5 GiB
  RAM and 4 x Nvidia A100 GPUs with 40 GiB RAM
- __Parameter for sbatch/salloc__:
    - `--partition=accel`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with `N` being the number of GPUs
- __Job Scripts__: {ref}`job_scripts_betzy_accel`

Can be combined with `--qos=devel` for shorter development tasks which require
GPUs for testing.

Note that *accel* jobs on Betzy are billed differently than *normal* jobs.
See the {ref}`accounting page<projects-accounting>` for more information.


(job_type_betzy_preproc)=

## Preproc

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 128 billing units (CPU cores plus memory) per job
    - maximum 1 node per job
    - maximum 16 running jobs per user
    - in total maximum 256 billing units in running jobs per user
- __Maximum walltime__: 1 day
- __Priority__: normal
- __Available resources__: 6 nodes, each with 128 CPU cores and 1 TiB RAM
- __Parameter for sbatch/salloc__:
    - `--qos=preproc`
- __Job Scripts__: {ref}`job_scripts_betzy_preproc`

*preproc* jobs are meant for small preprocessing or postprocessing
tasks.  Typically, such jobs don't use many CPUs, so requiring them to
use 4 whole nodes would waste resources.

Note that *preproc* jobs on Betzy are billed differently than *normal* jobs.
The details about how the billing units are calculated can be found
in [job accounting](../projects_accounting.md).


(job_type_betzy_devel)=

## Devel

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 node, maximum 4 nodes per job
    - maximum 1 running job per user
- __Maximum walltime__: 60 minutes
- __Priority__: high
- __Available resources__: 4 nodes with 128 CPU cores and 244 GiB RAM
- __Parameter for sbatch/salloc__:
    - `--qos=devel`
- __Job Scripts__: {ref}`job_scripts_betzy_devel`

This is meant for small, short development or test jobs.

Can be combined with `--partition=accel` to increase priority while
having max wall time and job limits of _devel_ job.

If you have _temporary_ development needs that cannot be fulfilled by the _devel_ job type, please contact us at
[support@nris.no](mailto:support@nris.no).
