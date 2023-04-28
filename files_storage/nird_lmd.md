(nird)=


# NIRD 
## National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
  is owned by [Sigma2](https://www.sigma2.no) and operated by [NRIS](https://www.sigma2.no/nris).

```{note}
The NIRD infrastructure offers storage services, archiving services, and
processing capacity for computing on the stored data.  It offers services
and capacities to any scientific discipline that requires access to
advanced, large scale, or high-end resources for storing, processing,
publishing research data or searching digital databases and collections.
```

The next generation NIRD storage system is installed in [Lefdal Mine Datacenter](https://www.sigma2.no/data-centre-facility). 
The new NIRD  is redesigned for the evolving needs of Norwegian researchers and has 
been procured through [the NIRD2020 project](https://www.sigma2.no/procurement-project-nird2020).


NIRD will provide storage resources with yearly capacity upgrades,
data security through backup services and adaptable application services,
 multiple storage protocol support, migration to third-party cloud 
providers and much more. Alongside the national high-performance computing 
resources, NIRD forms the backbone of the national e-infrastructure for research
and education in Norway, connecting data and computing resources for
efficient provisioning of services.

The **NIRD @LMD** architecture is based on IBM Elastic Storage System (ESS3200 & ESS5000 SC models) 
and consists of two separate storage systems, namely Tiered Storage (NIRD TS) and Data Lake (NIRD DL).

NIRD TS has two tiers, namely, performance and capacity tiers, while NIRD DL provides a unified access, i.e., file and object storage on the data lake.

The total capacity of the system is 33 PB (22 PB on NIRD TS and 11 PB on NIRD DL). 

The solution is based on IBM Spectrum Scale – an IBM Systems storage product based on GPFS (General Parallel File System). The solution provides a single storage architecture which is partitioned dynamically to provide for storage allocation for POSIX based file-services and S3 based object storage.

```{eval-rst}
.. toctree::
   :maxdepth: 1

   nird/access_lmd.md
   nird/storage-areas_lmd.md
   nird/snapshots_lmd.md
   nird/backup_lmd.md
   nird/mounts_lmd.md
   nird/old_nird_access.md
```


