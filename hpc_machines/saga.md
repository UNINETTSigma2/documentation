(saga)=

# Saga

The supercomputer is named after the goddess in norse mythology associated with wisdom. Saga is also a term for the Icelandic epic prose literature. The supercomputer, placed at NTNU in Trondheim is designed to run both sequential and parallel workloads. It was made available to users right before the start of the 2019.2 period (October 2019).

Saga is provided by Hewlett Packard Enterprise and has a computational capacity of approximately 140 million CPU hours a year and a life expectancy of four year, until 2023.


## Technical details

### Main components

* 200 standard compute nodes with 40 cores and 192 GiB memory each
* 120 standard compute nodes with 52 cores and 192 GiB memory each
* 28 medium memory compute nodes, with 40 cores and 384 GiB of memory each
* 8 big memory nodes, with 3 TiB and 64 cores each
* 8 GPU nodes, with 4 NVIDIA GPUs and 2 CPUs with 24 cores and 384 GiB memory each
* 8 login and service nodes with 256 cores in total
* 6.5 PB high metadata performance BeeGFS scratch file system

| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	16064  |
| Number of nodes     |	364  |
| Number of GPUs | 32 |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz / 6230R 2.1 GHz (normal)<br> Intel Xeon-Gold 6130 2.1 GHz (bigmem)<br> Intel Xeon-Gold 6126 2.6 GHz (accel)  |
| GPU type     |    NVIDIA P100, 16 GiB RAM (accel) |
| Total max floating point performance, double     |	645 Teraflop/s (CPUs) + 150 Teraflop/s (GPUs) |
| Total memory     |	97.5 TiB  |
| Total NVMe+SSD local disc | 89 TiB + 60 TiB |
| Total parallel filesystem capacity     |	1 PB  |


### Mapping of processors to memory specifications:

- 6126
  - HPE 16GB (1 x 16GB) Dual Rank x8 DDR4-2666 CAS-19-19-19 Registered Smart Memory Kit
  - Max number of memory channels: 6
- 6130
  - HPE 32GB (1 x 32GB) Dual Rank x4 DDR4-2666 CAS-19-19-19 Registered Smart Memory Kit
  - Max number of memory channels: 6
- 6138
  - HPE 64GB (1 x 64GB) Quad Rank x4 DDR4-2666 CAS-19-19-19 Load Reduced Smart Memory Kit
  - Max number of memory channels: 6
- 6230R
  - HPE 16GB (1 x 16GB) Dual Rank x8 DDR4-2933 CAS-21-21-21 Registered Smart Memory Kit
  - Max number of memory channels: 6
