(nird)=

# NIRD - National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
 is owned and operated by [Sigma2](https://www.sigma2.no).

```{warning}
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

 We have updated the new [NIRD documentation](https://documentation.sigma2.no/files_storage/nird_lmd.html). Please read the documentation carefully and revisit on a regular basis 
 for updated information. 
```


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
