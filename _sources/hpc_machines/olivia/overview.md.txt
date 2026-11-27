(olivia overview)=

# Olivia System architecture and technical specification
          
Olivia is based on HPE Cray EX platforms for compute and accelerator nodes, and HPE ClusterStor for global storage, all connected via HPE Slingshot high-speed interconnect.
          
## CPU Compute Nodes
        
The CPU-based compute resources consist of **252 nodes**. Each node features two **AMD EPYC Turin 128-core CPUs (Zen5c architecture)**, with **768 GiB of DDR5-6000 memory (24x32GB DIMMs)**. Each node also includes **3.84 TB of solid-state local storage (M.2 SSD)**. All Class A nodes are **direct liquid cooled**.

## Accelerator Nodes
    
These nodes are optimized for accelerated HPC and AI/ML workloads, totaling **76 nodes with 304 NVIDIA GH200 superchips**. Each GH200 device has **96 GB HBM3 memory (GPU)** and **120 GB LPDDR5x memory (CPU)**, connected via a **900 GB/s bidirectional NVlink C2C connection**. These nodes are also **direct liquid cooled**. Their dedicated storage is provided by a **HPE Cray C500 flash-based parallel filesystem with a total capacity of 510 TB**.
    
    
## Service Nodes
    
Olivia has **8 service nodes**, with 4 nodes having both CPU and GPU (**NVIDIA L40 GPUs** for remote visualization and compilation). Each node has **2 AMD EPYC 9534 64-core processors** and **1536 GiB DDR5-4800 memory**. Storage per node includes **2x 960 GB NVMe SSDs for the system drive** (configured as RAID mirror for redundancy) and **4x 6.4 TB NVMe SSDs for data (19.2 TB total)**. These nodes are **air-cooled**.

## I/O Nodes

Olivia has **5 dedicated nodes for I/O operations**, based on HPE ProLiant DL385 Gen 11. Each node has **2 AMD EPYC Genoa 9534 64-core CPUs** and **512 GiB memory**. They are designed to connect to the NIRD storage system and the Global Storage simultaneously. These nodes are **air-cooled**.

## High-Performance Global Storage

This system, based on ClusterStor E1000, offers **4.266 PB of total usable capacity**, including **1.107 PB of solid-state storage (NVMe)** and **3.159 PB of HDD-based storage**. It is POSIX compatible and can be divided into several file systems (e.g., scratch, home, software). It supports RDMA and is designed for fault tolerance against single component failures. The system is **air-cooled**.

## High-Speed Interconnect

The interconnect uses **HPE Slingshot**, a Dragonfly topology that supports **200 Gbit/s simplex bandwidth** between nodes. It natively supports IP over Ethernet, and RDMA or RoCE access to the storage solution.





