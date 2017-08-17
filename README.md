#HPC Resources

The current Norwegian academic HPC infrastructure consists of four systems, located in Tromsø ([Stallo](http://hpc-uit.readthedocs.io/en/latest/help/faq.html), Fram), and Oslo (Abel).

Each of the facilities consists of a compute resource (a number of compute nodes each with a number of processors and internal shared-memory, plus an interconnect that connects the nodes), a central storage resource that is accessible by all the nodes, and a secondary storage resource for back-up (and in few cases also for archiving). All facilities use variants of the UNIX operating system (Linux, AIX, etc.).

## Comparison between current hardware

The table below attempts to compare the available systems with respect to the type of applications they are suited for.

|Resource |	Job types |	Memory |	Cores/Node |
| :------------- | :------------- | :------------- | :------------- |
| Abel |	P   S   L |	60/1024 |	16/32 |
| Stallo |	P   S   L |	32/128 |	16/20 |
| Fram |	P&ensp;&ensp;&ensp;L |	64/512 |	32 |

**P** means the resource supports parallel jobs, **S** means the system supports serial jobs, and **L** means that large I/O jobs are supported.
The **Memory** column specifies physical node memory in Gigabytes (GiB). Abel and Stallo provides a few large memory nodes.
The **Cores** column is the number of physical cores in each node.

The resource allocation committee (RFK) manages a part of the total cores on the resources for national allocations. The national share is 8723 cores on Abel, 13796 cores on Stallo, and approximately 31600 cores on Fram.

The following considerations should be kept in mind when selecting a system to execute applications:

* Abel and Stallo are throughput systems. These systems can be used for sequential (single-threaded) as well as parallel applications.

* Fram is a system for large scale parallel (distributed-memory) applications. Applications that use less than 128 cores (or 32 nodes) are discouraged. Requests for access to execute applications that use fewer cores are often rejected or moved to other systems.

* Abel, Stallo and Fram runs CentOS Linux distributions. In case you need to install a specific software package, please make sure that you know for which environments the software is supported, before choosing a system.


## Developing HPC Applications

The following give new users information about running applications on the HPC machines:

* [Getting Started](quick/gettingstarted.md)
* [Porting from PBS/TORQUE to Slurm](jobs/porting.md)
* [Fram Setup](quick/fram.md)
* [Abel Documentation](http://www.uio.no/english/services/it/research/hpc/abel/)
* [Stallo Documentation](https://hpc-uit.readthedocs.io)


## NIRD Storage

Fram uses the NIRD storage system for storing archives for other research data. NOTUR projects have access
to this geo-replicated storage through various methods. Visit the [NIRD](storage/nird.md) page for more information.

### About UNINETT Sigma2

UNINETT Sigma2 AS manages the national infrastructure for computational science in Norway, and offers services in high performance computing and data storage.
Visit http://sigma2.no for more information.
