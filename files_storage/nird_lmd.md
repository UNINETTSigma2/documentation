(nird@lmd)=


# NIRD @LMD
## National Infrastructure for Research Data @Lefdal Mine Datacenter

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
 is owned and operated by [Sigma2](https://www.sigma2.no).

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

```{note}
 The system is under pre-production during which pilot users test the new infrastructure
 and data is migrated. The system will be online and accessible to all users in February 2023.
```

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
```


```{note}
We kindly remind you that the data migration from old NIRD to new NIRD is on-going and to avoid data loss or corruption we ask you to put your new/altered files during the pilot phase under the `/nird/projects/NSXXXK/_PILOT` folder.
```

## Backup and data integrity

```{warning}
Please note that data outside of the above mentioned `_PILOT` folder might be overwritten at any time. At the moment, until data migration is finished and transition to the new infrastructure is finalized, data integrity is ensured on storage level with erasure coding. 

Backups are currently not enabled. This is expected to be activated close to the production phase.`
```

Snapshots are activated on the new NIRD, it will allow data restoration in case of accidental deletion.

If you need increased redundancy in form of an extra copy of the dataset in a different location, you need to request this as an extra service. Should you have chosen replication or mixed replication in your application for storage resources for 2022.2, backup service will be switched on by the POWG team. POWG will contact each pilot project to guide setting up the inclusion/exclusion rule for each dataset. An inclusion/exclusion template will be provided later for all projects.


## NIRD Toolkit

Pilot users can test [NIRD Toolkit](https://documentation.sigma2.no/nird_toolkit/overview.html) on new NIRD. See the 
[NIRD Toolkit Documentation](https://documentation.sigma2.no/nird_toolkit/getting_started_guide.html) on how to deploy 
services on the toolkit. 
On the new nird, toolkit is available [here](https://store.sigma2.no). Begin by navigating to the package library in the 
[toolkit](https://store.sigma2.no).


```{warning}
- Before testing the nird toolkit on new nird, pilot users are requested to contact the POWG team at nird-migration@nris.no 
so that we can create dedicated namespaces/resources for the pilot project on the service platform at NIRD@LMD

- We do not have https certificates in place for the nird-lmd subdomain so users will have to "ignore" the certificate 
warning in their browser.

- Kindly remind that this is a pilot phase to test the toolkit, so services may experience unexpected downtimes.
```
