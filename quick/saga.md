# Saga

In norse mythology [Saga](https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr) is the goddess associated with wisdom.
The new Linux cluster hosted at [Norwegian University of Science and Technology](https://www.ntnu.edu)
(NTNU) is a shared resource for research computing capable of 645 TFLOP/s
theoretical peak performance. It entered full production during fall 2019.

Saga is a distributed memory system which consists of 244 dual/quad socket nodes,
interconnected with a high-bandwidth low-latency InfiniBand
network. Saga contains 200 `normal`, 36 `bigmem` and 8 `accel` compute nodes.
Each normal compute node has two 20-core Intel Skylake
chips (2.0 GHz), 192 GiB memory (186 GiB available for jobs) and a fast local NVMe disk (ca 300 GB shared by jobs).
There are two kinds of bigmem compute nodes: 28 contain two 20-core Intel Skylake chips (2.0 GHz), 384 GiB memory
(377 GiB available for jobs) and a fast local NVMe disk (ca 300 GB shared by jobs); 8 contain four 16-core Intel
Skylake chips (2.1 GHz) and 3072 GiB memory (3021 GiB available for jobs). Each GPU node
contains two 12-core Intel Skylake chips (2.6 GHz), 384 GiB memory (377 GiB available for jobs), 4 NVIDIA P100 GPUs
and fast local SSD-based disk RAID (about 8 TB in RAID1).

The total number of compute cores is 9824. Total memory is 75 TiB.


| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	9824  |
| Number of nodes     |	244  |
| Number of GPUs | 32 |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz (normal)<br> Intel Xeon-Gold 6130 2.1 GHz (bigmem)<br> Intel Xeon-Gold 6126 2.6 GHz (accel)  |
| GPU type     |    NVIDIA P100, 16 GiB RAM (accel) |
| Total max floating point performance, double     |	645 Teraflop/s (CPUs) + 150 Teraflop/s (GPUs) |
| Total memory     |	75 TiB  |
| Total NVMe+SSD local disc | 89 TiB + 60 TiB |
| Total parallel filesystem capacity     |	1 PB  |

The following give new users information about running applications on Fram:

* [Getting Started](gettingstarted.md)
* [Migrating to Saga](../faq/migration2saga.md)

More information on how to use Saga
can be found in the existing sections of the documentation in the left
hand menue, e.g, SOFTWARE, CODE DEVELOPMENT, JOBS, FILES AND STORAGE, etc.
