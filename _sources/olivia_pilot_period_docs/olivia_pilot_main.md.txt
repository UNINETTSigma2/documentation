---
orphan: true
---


(olivia_pilot_main)=

# Olivia Pilot Info

<!-- **Olivia User Guide ** (Skeleton draft) -->

```{danger}
__!! This page is currently under development and work in progress. Configs/information may not be final !!__
```

## Duration of pilot period
The pilot period is expected to start on Monday 7 July and will last until 30 September 2025.

## How to connect to Olivia
Logging into Olivia involves the use of {ref}`Secure Shell (SSH) <ssh>` protocol,
either in a terminal shell or through a graphical tool using this protocol
under the hood.  

In the future, we will introduce the same 2 factor authentication (2FA) system
as used on our other HPC machines. But for now, you cannot connect directly
from your local machine. Instead you have to __first connect to Betzy, Fram or
Saga and then further to Olivia__.

Replace `<username>` with your registered username:

```console
$ ssh <username>@betzy.sigma2.no
$ ssh <username>@olivia.sigma2.no
```

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

The calculation of elapsed GPU-hours is currently only dependent on the number of GPUs allocated for a job.


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

| Name   | Description       | Job limits | Max walltime | Priority |
|:------:|-------------------|:----------:|:------------:|:--------:|
| normal | default job type  |            | 7 days       | normal   |
| accel  | jobs needing GPUs |            | 7 days       | normal   |
| devel  | development jobs  |            | 2 hours      | normal   |

Note that job limits will change after pilot period is over.

#### Normal jobs (CPUs)

- __Allocation units__: CPUs and memory
- __Job Limits__:
    - no limits
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
	- 252 nodes each with 2*128 CPUs and 768 GiB RAM
- __Parameter for sbatch/salloc__:
    - None, _normal_ is the default
- Other notes:
    - This is the default job type.
    - Normal jobs have $SCRATCH on local NVMe disk on the nodes

(job_scripts_olivia_normal)=
##### Normal job script 
Job scripts for normal jobs on Olivia are very similar to the ones on
Saga.  For a job simple job running one process with 32 threads, the
following example is enough:

    #SBATCH --account=nnXXXXk     # Your project
    #SBATCH --job-name=MyJob      # Help you identify your jobs
    #SBATCH --time=1-0:0:0        # Total maximum runtime. In this case 1 day
    #SBATCH --cpus-per-task=32    # Number of CPU cores
    #SBATCH --mem-per-cpu=3G      # Amount of CPU memory per CPU core


#### Accel jobs (GPU)

*accel* jobs give access to use the H200 GPUs.

- __Allocation units__: CPUs, memory and GPUs
- __Job Limits__:
    - no limits
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes each with 4*72 ARM CPU cores, 4*120 GiB CPU RAM and 4*H200 GPUs (each with 96 GiB GPU memory).
- __Parameter for sbatch/salloc__:
    - `--partition=accel`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of GPUs
- __Other notes__:
    - Accel nodes have $SCRATCH on a shared flash storage file system.

(job_scripts_olivia_accel)=
##### Accel job script 
_Accel_ jobs are specified just like *normal* jobs except that they also have
to specify `--partition=accel` and the number of GPUs to use.  You can also
specify how the GPUs should be distributed across nodes and tasks.  The
simplest way to do that is, with `--gpus=N` or `--gpus-per-node=N`, where `N`
is the number of GPUs to use.

For a job simple job running one process and using one H200 GPU, the
following example is enough:

    #SBATCH --account=nnXXXXk     # Your project
    #SBATCH --job-name=MyJob      # Help you identify your jobs
    #SBATCH --partition=accel
    #SBATCH --gpus=1              # Total number of GPUs (incl. all memory of that GPU)
    #SBATCH --time=1-0:0:0        # Total maximum runtime. In this case 1 day
    #SBATCH --cpus-per-task=72    # All CPU cores of one Grace-Hopper card
    #SBATCH --mem-per-gpu=100G    # Amount of CPU memory

There are other GPU related specifications that can be used, and that
parallel some of the CPU related specifications.  The most useful are
probably:

- `--gpus-per-node` How many GPUs the job should have on each node.
- `--gpus-per-task` How many GPUs the job should have per task.
  Requires the use of `--ntasks` or `--gpus`.
- `--mem-per-gpu` How much (CPU) RAM the job should have for each GPU.
  Can be used *instead of* `--mem-per-cpu`, (but cannot be used
  *together with* it).

See [sbatch](https://slurm.schedmd.com/sbatch.html) or `man sbatch`
for the details, and other GPU related specifications.

### Interactive jobs
For technical reasons, interactive jobs started with `salloc` will for
the time being *not* start a shell on the first allocated compute node
of the job.  Instead, a shell is started on the *login node* where you
ran `salloc`.  (This will be changed before the pilot period is over.)

This means that the commands are run on the login node, not in the
compute node.  To run a command on the compute nodes in the
interactive job, either use `srun <command>` or `ssh` into the node
and run it there.


### Performance analysis tools


## Storage
Home directories and project directories are found in the usual place:
`/cluster/home` and `/cluster/projects`.  These have disk quotas; see
`dusage` for usage.

On Olivia, there is no personal work areas
(`/cluster/work/users/<uname>`).  Instead, there are *project* work
areas, one for each project, in `/cluster/work/projects/`.  This makes
it easier to share temporary files within the projects.  There is no
disk quotas in the project work areas.  Later, old files will be
deleted automatically, just like in the user work areas on the other
clusters.



## Debugging
## Data Storage/access(NIRD) and Transfer
## Guides to use Olivia effectively
### Best Practices on Olivia
<!-- OWS guide -->
## Any monitoring tools for users?
## User stories
## Troubleshooting
## Relevant tutorials



