---
orphan: true
---


(olivia_pilot_main)=

# Olivia pilot period start page (work in progress)

**Olivia User Guide ** (Skeleton draft)


## Architecture (Hardware)

Olivia is primarily based on HPE Cray EX platforms for compute and accelerator nodes, and HPE ClusterStor for global storage, all connected via HPE Slingshot high-speed interconnect.

### CPU Compute Nodes

These are the main CPU-based compute resources, consisting of **252 nodes**. Each node features two **AMD EPYC Turin 128-core CPUs (Zen5c architecture)**, with **768 GiB of DDR5-6000 memory (24x32GB DIMMs)**. Each node also includes **3.84 TB of solid-state local storage (M.2 SSD)**. All Class A nodes are **direct liquid cooled**.

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


## Project accounting (can be part of general accounting)
## Software

### Programming Environment (Based on the software implementation on CPE (EESSI/Module collections )
### Supported applications and libraries
### Containers?
### Software Installation as a user?


## Running Jobs

### Job types
### Example job scripts
### Performance analysis tools


## Debugging
## Data Storage/access(NIRD) and Transfer
## Best Practices on Olivia
## Guides to use Olivia effectively
## Any monitoring tools for users?
## User stories
## Trouble shooting
## Relevant tutorials



