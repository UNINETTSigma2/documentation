(nird)=


# NIRD 
## National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
  is owned by [Sigma2](https://www.sigma2.no) and operated by [NRIS](https://www.sigma2.no/nris).

```{note}
NIRD offers [storage services](https://www.sigma2.no/data-storage), [archiving services](https://www.sigma2.no/research-data-archive), [cloud services](https://www.sigma2.no/nird-service-platform) and processing capabilities on the stored data.  It offers services
and capacities to any scientific discipline that requires access to
advanced, large scale, or high-end resources for storing, processing,
publishing research data or searching digital databases and collections.

NIRD is a high-performance storage system, capable of supporting AI and analytics workloads, offering simultaneous multi-protocol access to the same data.
```

The next generation NIRD storage system is installed in [Lefdal Mine Datacenter](https://www.sigma2.no/data-centre-facility). 
The new NIRD  is redesigned for the evolving needs of Norwegian researchers and has 
been procured through [the NIRD2020 project](https://www.sigma2.no/procurement-project-nird2020).


NIRD provides storage resources with yearly capacity upgrades, data security through backup services and adaptable application services,
 multiple storage protocol support, migration to third-party cloud providers and much more. Alongside the national high-performance computing resources, NIRD forms the backbone of the national e-infrastructure for research and education in Norway, connecting data and computing resources for efficient provisioning of services.
 

### Technical Specifications


#### Hardware
**NIRD** consists of two separate storage systems, namely Tiered Storage (NIRD TS) and Data Lake (NIRD DL). The total capacity of the system is 49 PB (24 PB on NIRD TS and 25 PB on NIRD DL).

NIRD TS has several tiers spanned by single filesystem and designed for performance and used mainly for active project data.

NIRD DL has a flat structure, designed mainly for less active data. NIRD DL provides a unified access, i.e., file- and object storage for sharing data across multiple projects, and interfacing with external storages.

NIRD is based on IBM Elastic Storage System, built using ESS3200, ESS3500 and ESS5000 building blocks. I/O performance is ensured with IBM POWER servers for I/O operations, having dedicated data movers, protocol nodes and more.

| NIRD    | | |
| :------------- | :------------- | :------------- |
| System     |Building blocks |IBM ESS3200<br>IBM ESS3500<br>IBM ESS5000<br>IBM POWER9  |
| Clusters     |	Two physically separated clusters | NIRD TS<br>NIRD DL  |
| Storage media | NIRD TS<br>NIRD DL | NVMe SSD & NL-SAS<br>NL-SAS
| Capacity     |	Total capacity: 49 PB | NIRD TS: 24 PB<br> NIRD DL: 25 PB  |
| Performance | Aggregated I/O throughput | NIRD TS: 209 GB/s<br>NIRD DL: 66 GB/s |
| Interconnect | 100 Gbit/s Ethernet | NIRD TS: balanced 400 Gbit/s<br>NIRD DL: balanced 200 Gbit/s |
| Protocol nodes | NFS<br>S3 | 4 x 200 Gbit/s<br>5 x 50 Gbit/s|



#### Software
IBM Storage Scale (GPFS) is deployed on NIRD, providing a software-defined high-performance file- and object storage for AI and data intensive workloads.

Insight into data is ensured by IBM Storage Discover.

Backup services and data integrity is ensured with IBM Storage Protect. 

## In-depth documentation for NIRD

```{eval-rst}
.. toctree::
   :maxdepth: 1

   nird/access_lmd.md
   nird/storage-areas_lmd.md
   nird/snapshots_lmd.md
   nird/backup_lmd.md
   nird/mounts_lmd.md
   nird/ts_dl.md
   nird/faq_nird.md
   sharing_files.md   
```


