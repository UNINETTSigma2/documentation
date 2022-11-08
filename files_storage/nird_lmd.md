(nird@lmd)=


# NIRD@LMD - National Infrastructure for Research Data @ Lefdal Mine Datacenter

The next generation NIRD storage system is installed in [Lefdal Mine Datacenter](https://www.sigma2.no/data-centre-facility). The system is under pre-production during which pilot users test the new infrastructure and data is migrated. 

**We expect the system to be online and accessible to all users from December 2022.**

The new NIRD  is redesigned for the evolving needs of Norwegian researchers and has been procured through [the NIRD2020 project](https://www.sigma2.no/procurement-project-nird2020).


The **NIRD@LMD** architecture is based on IBM Elastic Storage System (ESS3200 & ESS5000 SC models) with storage tiers on tiered storage side and one unified datalake, providing object and file access. The total capacity of the system is 33 PB (22 PB on tiered storage and 11 PB on datalake).
The solution is based on IBM Spectrum Scale – an IBM Systems storage product based on GPFS (General Parallel File System). The solution provides a single storage architecture which is partitioned dynamically to provide for storage allocation for SMB, NFS and POSIX based file-services and S3 based object storage.


## Logging in 

The new nird can be accessed via the following address

```console
login.nird-lmd.sigma2.no
```

The login credentials are same as you use to login into the old nird.

You will be logged into your home area `/nird/home/$USERHOME`

```{note}
Your $HOME directory is  empty now. User home ($HOME) migration is user’s responsibility. The data migration from old nird to the new nird is ongoing now. However you can start tesing the new system. If you have any questions regarding the migration process and the configuration of the account on the new NIRD, do not hesitate to contact [the PoWG](https://www.sigma2.no/be-ready-migrate) via nird-migration@nris.no
```

## Project area

Each NIRD Data Storage project gets a project area either on tiered storage `/nird/projects/NSxxxxK` or on datalake `/nird/datalake/NSxxxxK` based on the project allocation.


## Backup

We take snapshots of the new NIRD to allow data restoration in case of accidental deletion. If you need increased redundancy in form of an extra copy of the dataset in a different location, you need to request this as an extra service. You will be also able to set up the inclusion/exclusion rule for each dataset yourself. 

## Quota 

You will keep the current quota on the old system until it is end of life. You get the updated quota on the new NIRD when launching in December. Thus, you have to count on the current quota until December.   


## TBA
