(nird)=


# NIRD
## National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
  is owned by [Sigma2](https://www.sigma2.no) and operated by [NRIS](https://www.sigma2.no/nris).

```{note}
NIRD offers storage services for [high-performance data analytics](https://www.sigma2.no/service/nird-data-peak), [unified file and object storage](https://www.sigma2.no/service/nird-data-lake), [archiving services](https://www.sigma2.no/research-data-archive), [cloud services](https://www.sigma2.no/nird-service-platform), and processing capabilities on the stored data.  It offers services
and capacities to any scientific discipline that requires access to
advanced, large scale, or high-end resources for storing, processing,
publishing research data or searching digital databases and collections.

NIRD is a high-performance storage system, capable of supporting AI and analytics workloads, offering simultaneous multi-protocol access to the same data.
```

The next generation NIRD storage system is installed in [Lefdal Mine Datacenter](https://www.sigma2.no/our-data-centre).
The new NIRD is redesigned for the evolving needs of Norwegian researchers and has
been procured through [the NIRD2020 project](https://www.sigma2.no/procurement-project-nird-2020).


NIRD provides storage resources with yearly capacity upgrades, data security
through backup services and adaptable application services, multiple storage
protocol support, migration to third-party cloud providers and much more.

Alongside the national high-performance computing resources, NIRD forms the
backbone of the national e-infrastructure for research and education in Norway,
connecting data and computing resources for efficient provisioning of services.


### Technical Specifications


#### Hardware
**NIRD** consists of two separate storage systems, namely **NIRD Data Peak** (codename *DP*) and **NIRD Data Lake** (codenamed *DL*), each tailored to optimally address two different categories of use cases. Commencing with the 2024.1 allocation, the array of functionalities provided by the DP and the DL resources are consolidated and presented as two distinct services. The total capacity of NIRD is 75 PB (24 PB on DP and 51 PB on DL).

**NIRD Data Peak** has several tiers spanned by single filesystem and designed for performance and used mainly for active project data.

**NIRD Data Lake** has a flat structure, designed mainly for less active data. The Data Lake provides a unified access, i.e., file- and object storage for sharing data across multiple projects, and interfacing with external storages.

NIRD is based on IBM Elastic Storage System, built using ESS3200, ESS3500 and ESS5000 building blocks. I/O performance is ensured with IBM POWER servers for I/O operations, having dedicated data movers, protocol nodes and more.

| NIRD    |                                   |                                                                                   |
| :------------- |:----------------------------------|:----------------------------------------------------------------------------------|
| System     | Building blocks                   | IBM ESS3200<br>IBM ESS3500<br>IBM ESS5000<br>IBM POWER9                           |
| Clusters     | 	Two physically separated clusters| NIRD DP<br>NIRD DL                                                                |
| Storage media | NIRD Data Peak<br>NIRD Data Lake  | NVMe SSD & NL-SAS<br>NL-SAS                                                       
| Capacity     | 	Total capacity: 75 PB              | NIRD Data Peak: 24 PB<br> NIRD Data Lake: 51 PB                                   |
| Performance | Aggregated I/O throughput         | NIRD Data Peak: 209 GB/s<br>NIRD Data Lake: 66 GB/s<br>NIRD Data Lake S3: 27 GB/s |
| Interconnect | 100 Gbit/s Ethernet               | NIRD Data Peak: balanced 400 Gbit/s<br>NIRD Data Lake: balanced 200 Gbit/s        |
| Protocol nodes | NFS<br>S3                         | 4 x 200 Gbit/s<br>5 x 50 Gbit/s                                                   |



#### Software
IBM Storage Scale (GPFS) is deployed on NIRD, providing a software-defined high-performance file- and object storage for AI and data intensive workloads.

Insight into data is ensured by IBM Storage Discover.

Backup services and data integrity is ensured with IBM Storage Protect.


## In-depth documentation for NIRD

```{toctree}
:maxdepth: 1

nird/access_lmd.md
nird/storage-areas_lmd.md
nird/s3.md
nird/snapshots_lmd.md
nird/backup_lmd.md
nird/mounts_lmd.md
nird/dp_dl.md
nird/faq_nird.md
```
