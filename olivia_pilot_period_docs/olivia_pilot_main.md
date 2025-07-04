---
orphan: true
---


(olivia_pilot_main)=

# Olivia pilot period start page (work in progress)

<!-- **Olivia User Guide ** (Skeleton draft) -->

# !! This page is currently under development and configs/information may not be final !!

## Duration of pilot period
The pilot period is expected to start on Monday 7 July and will last until 30 September 2025.

## How to connect to Olivia (This is not implemented yet)
Logging into Olivia involves the use of {ref}`Secure Shell (SSH) <ssh>` protocol,
either in a terminal shell or through a graphical tool using this protocol
under the hood.  SSH login is available natively to Linux or macOS. Also on
Windows a number of good tools for this exists.

Replace `<username>` with your registered username and `<machinename>` with the
specific machine name:
```console
$ ssh <username>@<machinename>
```
- `olivia.sigma2.no` - {ref}`olivia`


## System architecture and technical spec

Olivia is based on HPE Cray EX platforms for compute and accelerator nodes, and HPE ClusterStor for global storage, all connected via HPE Slingshot high-speed interconnect.

### CPU Compute Nodes

The CPU-based compute resources consist of **252 nodes**. Each node features two **AMD EPYC Turin 128-core CPUs (Zen5c architecture)**, with **768 GiB of DDR5-6000 memory (24x32GB DIMMs)**. Each node also includes **3.84 TB of solid-state local storage (M.2 SSD)**. All Class A nodes are **direct liquid cooled**.

### Accelerator Nodes

These nodes are optimized for accelerated HPC and AI/ML workloads, totaling **76 nodes with 304 NVIDIA GH200 superchips**. Each GH200 device has **96 GB HBM3 memory (GPU)** and **120 GB LPDDR5x memory (CPU)**, connected via a **900 GB/s bidirectional NVlink C2C connection**. These nodes are also **direct liquid cooled**. Their dedicated storage is provided by a **HPE Cray C500 flash-based parallel filesystem with a total capacity of 510 TB**.


### Service Nodes

Olivia has **8 service nodes**, with 4 nodes having both CPU and GPU (**NVIDIA L40 GPUs** for remote visualization and compilation). Each node has **2 AMD EPYC 9534 64-core processors** and **1536 GiB DDR5-4800 memory**. Storage per node includes **2x 960 GB NVMe SSDs for the system drive** (configured as RAID mirror for redundancy) and **4x 6.4 TB NVMe SSDs for data (19.2 TB total)**. These nodes are **air-cooled**.

### I/O Nodes

Olivia has **5 dedicated nodes for I/O operations**, based on HPE ProLiant DL385 Gen 11. Each node has **2 AMD EPYC Genoa 9534 64-core CPUs** and **512 GiB memory**. They are designed to connect to the NIRD storage system and the Global Storage simultaneously. These nodes are **air-cooled**.

### High-Performance Global Storage

This system, based on ClusterStor E1000, offers **4.266 PB of total usable capacity**, including **1.107 PB of solid-state storage (NVMe)** and **3.159 PB of HDD-based storage**. It is POSIX compatible and can be divided into several file systems (e.g., scratch, home, projects, software). It supports RDMA and is designed for fault tolerance against single component failures. The system is **air-cooled**.

### High-Speed Interconnect

The interconnect uses **HPE Slingshot**, a Dragonfly topology that supports **200 Gbit/s simplex bandwidth** between nodes. It natively supports IP over Ethernet, and RDMA or RoCE access to the storage solution.





## Project accounting (can be part of general accounting)
**Invoicing**: During the piloting phase there will no invoicing for usage of the system.

Project compute quota usage is accounted in 'billing units' BU. The calculation of elapsed BU for a job is done in the same manner as elsewhere on our systems.

The calculation of elapsed GPU-hours is currently only dependent on the amount of GPUs allocated for a job.


## Software

Olivia introduces several new aspects that set it apart from our current HPC systems. 

First and foremost, the large number of GPU nodes and the software that will run on them represent uncharted territory for us, as we don't yet have extensive experience in this area. Additionally, Olivia is an HPE Cray machine that leverages the [Cray Programming Environment (CPE)](https://cpe.ext.hpe.com/docs/latest/getting_started/CPE-General-User-Guide-HPCM.html), which comes with its own suite of tools for development, compilation, and debugging. As a result, we cannot simply replicate the software stack from our other systems. Instead, we need to adapt, modify, and expand the ways in which software is installed and utilized.

The pilot phase will serve as a testing ground for the approaches outlined below, helping us identify the best solutions for providing software on Olivia. Please note that __things will evolve during the pilot phase__, and the final configuration may differ significantly from what is described here. Your feedback is invaluable—if something doesn't work or could be improved, let us know so we can make adjustments.

### Software Setup

#### 1. Module System
We will continue to offer a wide range of software packages and libraries through the module system, similar to what is available on Saga, Fram, and Betzy. However, Olivia's architecture introduces a key difference: its CPU nodes use x86 processors, while its GPU nodes are based on ARM processors. To accommodate this, we will implement a hierarchical module system that displays only the relevant packages for the selected architecture. More details on this will be shared soon.

#### 2. Python, R, and (Ana-)Conda
Python and R, while widely used, were originally designed for personal computers rather than high-performance computing environments. These languages often involve installations with a large number of small files—Python environments, for example, can easily consist of tens or even hundreds of thousands of files. This can strain the file system, leading to poor performance and a sluggish user experience.

To address this, we will use [_HPC-container-wrapper_](https://github.com/CSCfi/hpc-container-wrapper/), a tool designed to encapsulate installations within containers optimized for HPC systems. This tool creates a container for your Python or Conda environment, significantly reducing the number of files visible to the file system. It also generates executables, allowing you to run commands like `python` seamlessly, without needing to interact directly with the container.

This approach minimizes file system load while maintaining ease of use. Detailed documentation on how to install and manage your software using _HPC-container-wrapper_ will be provided soon.

#### 3. AI Frameworks
For AI workflows based on popular frameworks like PyTorch, JAX, or TensorFlow, we aim to deliver optimal performance by utilizing containers provided by [Nvidia](https://catalog.ngc.nvidia.com/containers). These containers are specifically optimized for GPU workloads and should provide excellent performance while remaining relatively straightforward to use.

At the start of the pilot phase, you will need to run these containers directly using Singularity. We will soon provide comprehensive documentation to guide you through this process. Later, we plan to simplify the experience by integrating these tools into the module system. This will allow you to load a module that provides executables like `python` and `torch`, eliminating the need to interact with the containers directly.



## Running Jobs

### Job types

For the pilot period of Olivia, the following job types are defined.

| Name                                    | Description                               | Job limits   | Max walltime | Priority |
|:---------------------------------------:|-------------------------------------------|:------------:|:------------:|:--------:|
| normal   | default job type                          | 1--2048 units | 7 days       | normal   |
| accel     | jobs needing GPUs                         | 0-$\infin$ GPUs, 1-$\infin$ CPUs             | 7 days       | normal   |
| dev-g (TBD - noe til prod)     | inspired by LUMI?                         |             | max 2 hours       | normal   |

See projects-accounting for how the units are calculated.

#### Normal jobs (cpu)

- __Allocation units__: cpus and memory
- __Job Limits__:
    - maximum 2048 units
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
	- 252 nodes each with 2*128 cpus and 768 GiB RAM
- __Parameter for sbatch/salloc__:
    - None, _normal_ is the default
<!-- - __Job Scripts__: {ref}`job_scripts_saga_normal` -->

This is the default job type.  Most jobs are *normal* jobs.

Normal jobs will utilize local NVMe storage as default scratch location.


#### Accel jobs (gpu)

- __Allocation units__: cpus, memory and GPUs
- __Job Limits__:
    - no limits
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes each with 2*72 arm cpus, 240 GiB CPU RAM and 2 H100 96GB
  GPUs.
- __Parameter for sbatch/salloc__:
    - `--partition=accel`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of GPUs
- __Job Scripts__: {ref}`job_scripts_saga_accel`

*accel* jobs give access to use the H100 GPUs.

Accel nodes will utilize the global file system with flash storage as default scratch location.

### Example job scripts
### Performance analysis tools

## Storage



## Debugging
## Data Storage/access(NIRD) and Transfer
## Guides to use Olivia effectively
### Best Practices on Olivia
<!-- OWS guide -->
## Any monitoring tools for users?
## User stories
## Troubleshooting
## Relevant tutorials



