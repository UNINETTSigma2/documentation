(nird@lmd)=


# NIRD @LMD - National Infrastructure for Research Data @Lefdal Mine Datacenter

The next generation NIRD storage system is installed in [Lefdal Mine Datacenter](https://www.sigma2.no/data-centre-facility). The system is under pre-production during which pilot users test the new infrastructure and data is migrated. 

**We expect the system to be online and accessible to all users late December 2022 / January 2023.**

The new NIRD  is redesigned for the evolving needs of Norwegian researchers and has been procured through [the NIRD2020 project](https://www.sigma2.no/procurement-project-nird2020).


The **NIRD @LMD** architecture is based on IBM Elastic Storage System (ESS3200 & ESS5000 SC models) and consists of two separate storage systems, namely Tiered Storage (NIRD TS) and Data Lake (NIRD DL).

NIRD TS has two tiers, namely, performance and capacity tiers, while NIRD DL provides a unified access, i.e., file- and object storage on the data lake.

The total capacity of the system is 33 PB (22 PB on NIRD TS and 11 PB on NIRD DL).

The solution is based on IBM Spectrum Scale – an IBM Systems storage product based on GPFS (General Parallel File System). The solution provides a single storage architecture which is partitioned dynamically to provide for storage allocation for POSIX based file-services and S3 based object storage.


## Logging in 

The new NIRD can be accessed via the following address

```console
login.nird-lmd.sigma2.no
```

The login credentials are the same as for the old NIRD.

You will be logged into your home area `/nird/home/$USERHOME`

```{note}
Your $HOME directory is  empty now. User home ($HOME) migration is user’s responsibility.  The data migration from old NIRD to the new NIRD is on-going now. However, you can start tesing the new system. If you have any questions regarding the migration process and the configuration of the account on the new NIRD, do not hesitate to contact [the PoWG](https://www.sigma2.no/be-ready-migrate) via nird-migration@nris.no
```

## Project area

Each NIRD Data Storage project gets a project area either on NIRD TS `/nird/projects/NSxxxxK` or on NIRD DL `/nird/datalake/NSxxxxK` based on the project allocation.

```{note}
We kindly remind you that the data migration from old NIRD to new NIRD is on-going and to avoid data loss or corruption we ask you to put your new/altered files during the pilot phase under the `/nird/projects/NSXXXK/_PILOT` folder.
```

## Backup and data integrity

```{warning}
Please note that data outside of the above mentioned _PILOT folder might be overwritten at any time. At the moment, until data migration is finished and transition to the new infrastructure is finalized, data integrity is ensured on storage level with erasure coding. 

Backups are currently not enabled. This is expected to be activated close to the production phase.`
```

Snapshots are activated on the new NIRD, it will allow data restoration in case of accidental deletion.

If you need increased redundancy in form of an extra copy of the dataset in a different location, you need to request this as an extra service. Should you have chosen replication or mixed replication in your application for storage resources for 2022.2, backup service will be switched on by the POWG team. POWG will contact each pilot project to guide setting up the inclusion/exclusion rule for each dataset. An inclusion/exclusion template will be provided later for all projects.

## Quota 

You will keep the current quota on the old system until it is end of life. 
You get the quota allocated by RFK for the 2022.2 allocation period NIRD.

```{note}
`dusage` command on new nird are yet to be configured. We shall update the documentation when it is implemented.
```
## Software access

Software access is via command line. 
