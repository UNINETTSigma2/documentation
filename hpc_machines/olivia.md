---
orphan: true
---

(olivia)=


# Olivia

The name Olivia is inspired by ´olivine´, a mineral once extracted from the former Lefdal mine, now transformed into a state-of-the-art data centre, [Lefdal Mine Data Centers (LMD)](https://www.lefdalmine.com/), where the supercomputer is hosted. 

```{danger}
[Information for pilot users here.](/olivia_pilot_period_docs/olivia_pilot_main.md)
```

## The next generation powerful supercomputer for HPC and AI applications

Olivia is provided by Hewlett Packard Enterprise and will have a computational capacity 17 times greater than the current most powerful national supercomputer, Betzy. 

## Technical Details

| Details     | Olivia     |
| :------------- | :------------- |
| System     | HPE Cray Supercomputing EX  |
| Number of CPU nodes     |     252  |
| CPU type     |        AMD Epyc Turin  |
| CPU cores in total  | 64512 |
| CPU cores per node  | 256 |
| Memory in total    |  189 TiB  |
| Memory per node    |  768 GiB  |
| GPU type | NVIDIA GraceHopper Superchip (NVIDIA GH200 120GiB) |
| Number of GPU nodes | 76 |
| GPUs per node | 4 |
| GPUs in total | 304 |
| Interconnect  |  HPE Slingshot Interconnect | 

### Regular Compute nodes (HPC):

- 252 compute 2 way nodes with a total of 64512 AMD cores
- 768GiB memory per node, giving a total of 189 TiB
- Slingshot injection port 200 Gbits/s, with combined capacity of 49 Tbits/s
- 3.84 TB NVMe per node, yielding 967 TB for local scratch

### Accelerate nodes (AI):

- 76 nodes with GPUs,each node has four NVIDIA GraceHopper Superchip
   - 76 accelerated 4 way nodes with total of 21888(76*4*72) ARM cores
   - 76 accelerated nodes with a total of 304 (76*4) Hopper accelerators
- 384 GiB HMB3 accelerator memory per node total aggregated 28 TiB
- 480 GiB LPDDR5 processor memory per node total aggregated 35TiB
- Injection ports yielding 800 Gbits/s, giving combined bandwidth of 60 Tbits/s

### Cluster Storage:

- 3.159 PB spinning disks
- 1.107 PB solid state storage

## In-depth documentation for Olivia

```{toctree}
:maxdepth: 1

olivia/access.md
olivia/software_stack.md
olivia/jobs.md
olivia/ai_ml_guide.md
olivia/trouble_shooting.md
```

## [Olivia Best Practices Guide](xx)

