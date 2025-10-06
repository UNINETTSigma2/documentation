---
orphan: true
---
(job-types-olivia)=

# Job Types on Olivia

Olivia is designed for large-scale parallel jobs and GPU-accelerated workloads.
With its high-performance compute nodes featuring 256 or 288 CPUs and substantial
memory per node, Olivia is suited for computationally intensive
applications that can scale across many cores.

The basic allocation units on Olivia are cpus, memory and GPUs. The
details about how the billing units are calculated can be found in
{ref}`projects-accounting`.  Note that the number of GPUs is counted
separately, not as part of the billing units.

(job_type_olivia_normal)=

## Normal

- __Allocation units__: cpus and memory
- __Job Limits__:
  - maximum 1152 units
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
  - 252 nodes with 256 AMD cpus and 753 GiB RAM
- __Parameter for sbatch/salloc__:
  - None, _normal_ is the default
- __Job Scripts__: {ref}`job_scripts_olivia_normal`

This is the default job type. Most CPU-only jobs run as *normal* jobs. The large
node size (256 CPUs) makes this partition good for:

- Large-scale parallel computations
- Memory-intensive applications requiring substantial RAM
- Jobs that can efficiently utilize many CPU cores
- High-throughput computing workflows
- Scientific simulations requiring significant computational resources

(job_type_olivia_accel)=

## Accel

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
  - maximum 1152 billing units
  - maximum 32 GPUs
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes (max 60 per project) with 288
  ARM64 cpus, 858 GiB RAM and 4 GH200 GPUs.
- __Parameter for sbatch/salloc__:
  - `--partition=accel`
  - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of
    GPUs
- __Job Scripts__: {ref}`job_scripts_olivia_accel`

*Accel* jobs give access to use the Grace Hopper nodes that combine ARM64 CPUs with
NVIDIA GH200 GPUs.  This is useful for  AI/ML training, inference, and
other GPU-accelerated applications.

Can be combined with `--qos=devel` to get higher priority but maximum wall time (2h)
and resource limits of _devel_ apply.

(job_type_olivia_devel)=

## Devel

- __Allocation units__: cpus and memory and GPUs
- __Job Limits__:
    - maximum 576 billing units per job
    - maximum 32 GPUs per job
    - maximum 1152 billing units in use at the same time
    - maximum 64 GPUs in use at the same time
    - maximum 2 running jobs per user
- __Maximum walltime__: 2 hours
- __Priority__: high
- __Available resources__: *devel* jobs can run on any node on Olivia
- __Parameter for sbatch/salloc__:
    - `--qos=devel`
- __Job Scripts__: {ref}`job_scripts_olivia_devel`

This is meant for small, short development or test jobs.  *Devel* jobs
get higher priority for them to run as soon as possible.  On the other
hand, there are limits on the size and number of _devel_ jobs.

Can be combined with `--partition=accel` to increase
priority while having max wall time and job limits of _devel_ job.

If you have _temporary_ development needs that cannot be fulfilled by
the _devel_ or _short_ job types, please contact us at
[support@nris.no](mailto:support@nris.no).
