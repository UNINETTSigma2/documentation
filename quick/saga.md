# Saga

In norse mythology [Saga](https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr) is the goddess associated with wisdom.
The new Linux cluster is hosted at [Norwegian University of Science and Technology](https://www.ntnu.edu)
(NTNU) is a shared resource for research computing capable of 645 TFLOP/s
theoretical peak performance. It is scheduled to enter full production during fall 2019.

Saga is a distributed memory system which consists of 244 dual socket nodes,
interconnected with a high-bandwidth low-latency InfiniBand
network. Saga contains 200 standard, 28 mediummem, 8 largemem and 8 gpu compute nodes.
Each standard (mediummem) compute node has two 20-core Intel Skylake
chips (2.0 GHz) and 192 GiB memory (mediummem: 384 GiB). Each largemem compute node
contains four 16-core Intel Skylake chips (2.1 GHz) and 3072 GiB memory. GPU nodes
contain two 12-core Intel Skylake chips (2.6 GHz), 384 GiB memory and 4 NVIDIA P100 GPUs.

The total number of compute cores is 9824. Total memory is 75 TiB.


| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	9824  |
| Number of nodes     |	244  |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz (standard & mediummem)<br> Intel Xeon-Gold 6130 2.1 GHz (largemem)<br> Intel Xeon-Gold 6126 2.6 GHz (gpu)  |
| Max Floating point performance, double     |	645 Teraflop/s  |
| Total memory     |	75 TiB  |
| Total disc capacity     |	1 PB  |


More information on how to use Saga
will be added here when the system becomes ready for production.
