---
orphan: true
---
(job-types-olivia)=

# Job Types on Olivia

Olivia is designed for large-scale parallel jobs and GPU-accelerated workloads.
With its high-performance compute nodes featuring 256-288 CPUs and substantial
memory per node, Olivia is well-suited for computationally intensive
applications that can scale across many cores.

The basic allocation units on Olivia are cpu and memory. The details about how
the billing units are calculated can be found in {ref}`projects-accounting`.

Olivia offers two main types of compute resources:

## CPU-Intensive Workloads

For large-scale parallel computations, memory-intensive applications, and
CPU-bound tasks that can efficiently utilize many cores, Olivia provides
dedicated CPU nodes with exceptional compute density.

(job_type_olivia_normal)=

### Normal

- __Allocation units__: cpus and memory
- __Job Limits__:
  - No specific unit limits (cluster-dependent)
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
  - 252 nodes with 256 cpus and ~753 GiB RAM per node
- __Parameter for sbatch/salloc__:
  - None, _normal_ is the default
- __Job Scripts__: {ref}`job_scripts_olivia_normal`

This is the default job type. Most CPU-only jobs run as *normal* jobs. The large
node size (256 CPUs) makes this partition ideal for:

- Large-scale parallel computations
- Memory-intensive applications requiring substantial RAM
- Jobs that can efficiently utilize many CPU cores
- High-throughput computing workflows
- Scientific simulations requiring significant computational resources

## GPU-Intensive Workloads

For AI/ML training, inference, and other GPU-accelerated applications, Olivia
provides cutting-edge Grace Hopper nodes that combine powerful ARM64 CPUs with
advanced NVIDIA GH200 GPUs.

(job_type_olivia_accel)=

### Accel

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
  - No specific unit limits (cluster-dependent)
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes with 288 ARM64 cpus, ~858 GiB RAM and 4
  GH200 GPUs per node
- __Parameter for sbatch/salloc__:
  - `--partition=accel`
  - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of
    GPUs
- __Job Scripts__: {ref}`job_scripts_olivia_accel`
