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
- __Available resources__: 1340 nodes with 128 CPU cores and 250 GiB RAM
- __Parameter for sbatch/srun__:
    - None, _normal_ is the default

This is the default job type. In _normal_ jobs, the queue system hands out complete nodes.


(job_type_betzy_preproc)=

## Preproc

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 8 billing units (CPU cores plus memory) per job
    - maximum 2 running jobs per user
- __Maximum walltime__: 1 day
- __Priority__: normal
- __Available resources__: 128 CPU cores and 250 GiB RAM on one node
- __Parameter for sbatch/srun__:
    - `--qos=preproc`

*preproc* jobs are meant for small preprocessing or postprocessing
tasks.  Typically, such jobs don't use many CPUs, so requiring them to
use 8 whole nodes would waste resources.

The details about how the billing units are calculated can be found
in {ref}`job_accounting`.


(job_type_betzy_devel)=

## Devel

- __Allocation units__: whole nodes
- __Job Limits__:
    - minimum 1 node, maximum 4 nodes per job
    - maximum 1 running job per user
- __Maximum walltime__: 60 minutes
- __Priority__: high
- __Available resources__: 4 nodes with 128 CPU cores and 250 GiB RAM
- __Parameter for sbatch/srun__: 
    - `--qos=devel`

This is meant for small, short development or test jobs.  *Devel* jobs have access to a set of dedicated nodes.

If you have _temporary_ development needs that cannot be fulfilled by the _devel_ job type, please contact us at
[support@metacenter.no](mailto:support@metacenter.no).
