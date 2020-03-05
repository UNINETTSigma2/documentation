#HPC and Storage Documentation

The current Norwegian academic HPC infrastructure consists of three systems, located in Tromsø ([Stallo](http://hpc-uit.readthedocs.io/en/latest/help/faq.html), [Fram](quick/fram.md)) and Trondheim ([Saga](quick/saga.md)).

Each of the facilities consists of a compute resource (a number of compute nodes each with a number of processors and internal shared-memory, plus an interconnect that connects the nodes), a central storage resource that is accessible by all the nodes, and a secondary storage resource for back-up (and in few cases also for archiving). All facilities use variants of the UNIX operating system (Linux, AIX, etc.).

Additionally, [NIRD](storage/nird.md) provides storage and archiving services for research data.

## Developing HPC Applications

The following give new users information about running applications on the HPC machines:

* [Getting Started](quick/gettingstarted.md)
* [Porting from PBS/TORQUE to Slurm](jobs/porting.md)
* [Fram Setup](quick/fram.md)
* [Stallo Documentation](https://hpc-uit.readthedocs.io)
* [Saga Setup](quick/saga.md)


## NIRD Storage

Fram and Saga use the NIRD storage system for storing archives for other research data. NOTUR projects have access
to this geo-replicated storage through various methods. Visit the [NIRD](storage/nird.md) page for more information.

## Storage areas on Fram and Saga

You can find details about the storage systems available on Fram and Saga
[here](storage/clusters.md).

### About UNINETT Sigma2

UNINETT Sigma2 AS manages the national infrastructure for computational science in Norway, and offers services in high performance computing and data storage.
Visit https://www.sigma2.no for more information.

Latest news and announcements from Metacenter are posted at <a href="https://opslog.sigma2.no" target="_blank">Metacenter OpsLog</a> and the @MetacenterOps Twitter channel.
