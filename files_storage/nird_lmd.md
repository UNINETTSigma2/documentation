(nird@lmd)=


# NIRD @LMD
## National Infrastructure for Research Data @Lefdal Mine Datacenter

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

```{note}
 **IMPORTANT**: The new NIRD will open for users from 22.Feb.2023. Data has been migrated,
 but still, we need to do a final sync between the old and the new system before giving
 access to the new one. During this operation you will not have write access to the respective
 project neither on the old nor on the new NIRD. [Here](https://md.sigma2.no/NIRD-on-production?view) you can see when each project will have the final sync and switch over the new system.

 Please check the [list](https://md.sigma2.no/NIRD-on-production?view) and be informed on when you will no longer have access to the old NIRD.
 NIRD Project leaders are informed via email about the same.

 We kindly remind the users who have access to more than one project, to limit their activity
 on the new NIRD to the project(s) which were migrated, and access has been confirmed by
 the preparation for operation working group (POWG) team.

 Mount points for NIRD on the HPC systems and DNS entries will be updated as soon as all projects
 are migrated. Access to your $HOME folder will be available during migration on both old and new NIRD.
 You as users are now responsible for migrating the data to your $HOME folders by yourself when access
 will be given to you in the new NIRD.
 
 
 Pilot users who got access earlier are requested to follow the same workflow as earlier
 until your project gets the regular access. ie,  put your new/altered files under 
 the `/nird/projects/NSXXXK/_PILOT` to avoid data corruption. 

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
   nird/snapshots_lmd.md
   nird/backup_lmd.md



