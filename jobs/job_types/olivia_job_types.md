---
orphan: true
---
(job-types-olivia)=

# Job Types on Olivia

Olivia is designed for large-scale parallel jobs and GPU-accelerated workloads.
With its high-performance compute nodes featuring 256 or 288 CPUs and substantial
memory per node, Olivia is suited for computationally intensive
applications that can scale across many cores.

The basic allocation units on Olivia are cpus, memory and GPUs, or
whole nodes.  The
details about how the billing units are calculated can be found in
{ref}`projects-accounting`.  Note that the number of GPUs is counted
separately, not as part of the billing units.

(job_type_olivia_small)=

## Small

- __Allocation units__: cpus and memory
- __Job Limits__:
  - maximum 256 billing units
  - maximum 1 node
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
  - 88 nodes with 256 AMD cpus and 741 GiB RAM
- __Parameter for sbatch/salloc__:
  - `--partition=small` (can be omitted, _small_ is the default)
- __Job Scripts__: {ref}`job_scripts_olivia_small`

This is the default job type, meant for CPU-only jobs needing less
than a whole node.  The partition is good for:

- Memory-intensive applications requiring substantial RAM but less than
  256 CPUs.

(job_type_olivia_large)=

## Large

- __Allocation units__: whole nodes
- __Job Limits__:
  - maximum 9 nodes
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
  - 172 nodes with 256 AMD cpus and 741 GiB RAM
- __Parameter for sbatch/salloc__:
  - `--partition=large`
- __Job Scripts__: {ref}`job_scripts_olivia_large`

This is meant for larger CPU-only jobs, needing at least one node.
The partition is good for:

- Large-scale parallel computations
- Memory-intensive applications requiring substantial RAM and many CPUs
- Jobs that can efficiently utilize many CPU cores
- Scientific simulations requiring significant computational resources

Note that jobs will be allocated to whole nodes, no matter what
`--ntasks` and/or `--ntasks-per-node` are specified as.  A notice will
be printed about this at submission if needed.  This can be suppressed
by setting the environment variable
`SLURM_SUBMIT_SUPPRESS_NTASKS_WARNING` or
`SLURM_SUBMIT_SUPPRESS_WARNINGS` to `1` (the latter will suppress any
warnings from submission).

(job_type_olivia_accel)=

## Accel

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
  - minimum 1 GPU
  - maximum 32 GPUs
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes (max 60 per project) with 288
  ARM64 cpus, 808 GiB RAM and 4 GH200 GPUs.
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
    - maximum 1152 billing units per job
    - maximum 16 GPUs per job
    - maximum 2304 billing units in use at the same time
    - maximum 32 GPUs in use at the same time
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

Can be combined with `--partition=small`, `--partition=large` or
`--partition=accel` to increase priority while having max wall time
and job limits of _devel_ job.

If you have _temporary_ development needs that cannot be fulfilled by
the _devel_ job type, please contact us at
[support@nris.no](mailto:support@nris.no).
