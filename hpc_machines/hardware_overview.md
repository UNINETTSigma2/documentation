# Overview over our machines
The current Norwegian academic HPC infrastructure consists of three systems, located in Tromsø ([Fram](/hpc_machines/fram.md)) and Trondheim ([Saga](/hpc_machines/saga.md) and [Betzy](/hpc_machines/betzy.md)).

Each of the facilities consists of a compute resource (a number of compute nodes each with a number of processors and internal shared-memory, plus an interconnect that connects the nodes), a central storage resource that is accessible by all the nodes, and a secondary storage resource for back-up (and in few cases also for archiving). All facilities use variants of the UNIX operating system.

Additionally, [NIRD](/files_storage/nird.md) provides a service platform, storage and archiving services for research data. Here you can for instance request your application to be deployed, build integrated solutions and make sure data is safe and accessible.

## Comparison between current hardware

The table below compares the available systems with respect to the type of applications they are suited for. Second column indicates which job profile type is supported by the resource given in first column; **A** means GPU-accelerated jobs, **P** means parallel jobs, **S** means serial jobs, and **L** means large I/O jobs. 
The **Memory** column specifies physical node memory in Gigabytes (GiB), with minimum and maximum numbers given. There are uneven distritutions in the memory categories, please read specs for every given machine in order to find the exact numbers. The **Cores** column is the number of physical cores in each node.

|Resource |	Job types |	Memory (min/max) |	Cores/Node |
| :------------- | :------------- | :------------- | :------------- |
| Betzy |	P&ensp;&ensp;&ensp;L |	256/256 |	128 |
| Fram |	P&ensp;&ensp;&ensp;L |	64/512 |	32 |
| Saga |    A   P   S   L | 186/3066 |  24/64 |



The resource allocation committee (RFK) manages a part of the total cores on the resources for national allocations. The national share is 9824 cores on Saga, approximately 31600 cores on Fram and 172032 cores on Betzy.

The following considerations should be kept in mind when selecting a system to execute applications:

* Saga is a throughput system. It can be used for sequential (single-threaded) as well as parallel applications.

* Betzy and Fram are large scale parallel (distributed-memory) application systems.
Applications that use less than 128 cores (4 nodes) on Fram and less than 1024 cores (8 nodes) on Betzy for production are discouraged.
Requests for access to execute applications that use fewer cores are often rejected or moved to other systems.

* Saga and Fram run CentOS Linux distributions.

* Betzy runs Bull Super Computer Suite 5 (SCS5) based on RHEL 7™.

* In case you need to install a specific software package, please make sure that you know for which environments the software is supported, before choosing a system.
