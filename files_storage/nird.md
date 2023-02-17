(nird)=

# NIRD - National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
 is owned and operated by [Sigma2](https://www.sigma2.no).

```{note}
The NIRD infrastructure offers storage services, archiving services, and
processing capacity for computing on the stored data.  It offers services
and capacities to any scientific discipline that requires access to
advanced, large scale, or high-end resources for storing, processing,
publishing research data or searching digital databases and collections.
```

NIRD will provide storage resources with yearly capacity upgrades,
data security through geo-replication (data stored on two physical
locations) and adaptable application services, multiple storage
protocol support, migration to third-party cloud providers and much
more. Alongside the national high-performance computing resources,
NIRD forms the backbone of the national e-infrastructure for research
and education in Norway, connecting data and computing resources for
efficient provisioning of services.

The **NIRD storage system** consists of DDN SFA14K controllers, 3400 x 10TB NL-SAS
drives with a total capacity of 2 x 11 PiB.  The solution is based on DDN
GridScaler® parallel file system, supporting multiple file, block and object
protocols.

```{note}
**IMPORTANT**: The next generation NIRD is installed at Lefdal Mine Datacenter 
and starts production from 22.02.2023. Please see the [the second generation NIRD user guide](nird_lmd.md)
 for more details.
```

The **NIRD toolkit** allows pre/post processing analysis, data intensive
processing, visualization, artificial intelligence and machine learning
platform.  The NIRD toolkit services have access to your NIRD Project area.
The available services can be found at the documentation of [NIRD
Toolkit](https://www.sigma2.no/nird-toolkit) .

```{eval-rst}
.. toctree::
   :maxdepth: 1

   nird/access.md
   nird/storage-areas.md
   nird/replication.md
   nird/mounts.md
   nird/migration.md
```
