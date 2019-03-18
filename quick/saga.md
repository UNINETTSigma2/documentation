# Saga

*Note! Saga is scheduled to enter full production during fall 2019.*

In norse mythology [Saga](https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr) is the goddess associated with wisdom.
The new Linux cluster hosted at [Norwegian University of Science and Technology](https://www.ntnu.edu)
(NTNU) is a shared resource for research computing capable of 645 TFLOP/s
theoretical peak performance. It is scheduled to enter full production during fall 2019.

Saga is a distributed memory system which consists of 244 dual/quad socket nodes,
interconnected with a high-bandwidth low-latency InfiniBand
network. Saga contains 200 standard, 28 mediummem, 8 largemem and 8 gpu compute nodes.
Each standard (mediummem) compute node has two 20-core Intel Skylake
chips (2.0 GHz), 192 GiB memory (mediummem: 384 GiB) and a fast local NVMe disk (400 GB).
Each largemem compute node
contains four 16-core Intel Skylake chips (2.1 GHz) and 3072 GiB memory. Each GPU node
contains two 12-core Intel Skylake chips (2.6 GHz), 384 GiB memory, 4 NVIDIA P100 GPUs
and fast local SSD-based disk RAID (about 8 TB in RAID1).

The total number of compute cores is 9824. Total memory is 75 TiB.


| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	9824  |
| Number of nodes     |	244  |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz (standard & mediummem)<br> Intel Xeon-Gold 6130 2.1 GHz (largemem)<br> Intel Xeon-Gold 6126 2.6 GHz (gpu)  |
| Max Floating point performance, double     |	645 Teraflop/s  |
| Total memory     |	75 TiB  |
| Total NVMe+SSD local disc | 89 TiB + 60 TiB |
| Total number of GPUs | 32 NVIDIA P100, 16 GiB RAM |
| Total disc capacity     |	1 PB  |


More information on how to use Saga
will be added here when the system becomes ready for production.
