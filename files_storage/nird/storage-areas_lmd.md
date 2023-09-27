# Storage areas, quota, and backup


## Home directories

Each user has a home directory `/nird/home/<username>`, where
`<username>` is the username. The default quota for home directories
is 60 GiB and 300 000 files. To check the disk usage and quotas, type:

```console
$ dusage
```

Home directories on NIRD also contain a backup of Betzy, Fram and Saga home
directories (when relevant) in `/nird/home/<username>/backup/fram` , 
`/nird/home/<username>/backup/saga` and  `/nird/home/<username>/backup/betzy`.
The default quota is including the backup of HPC home directory. 


## Scratch directories

The total storage space of `/scratch` is 30 TiB.
Each user has a scratch directory `/nird/scratch/<username>`.
The area is meant as a temporary scratch area. This area is not backed up. 
There is no quota in the scratch area.

The `/scratch/` area enforces an automatic cleanup strategy, files in this 
area will be deleted after 21 days.
If file system usage reaches 75%, files in `/scratch/` will be deleted even 
before 21 days. 


## Project area

Each NIRD Data Storage project gets a project area either on NIRD TS `/nird/projects/NSxxxxK`
 or on NIRD DL `/nird/datalake/NSxxxxK` based on the project allocation,
 where `NSxxxxK` is the ID of the project.

The project area has a quota on disk space and the number of files.

Quotas are allocated by the Resource Allocation Committee (RFK) on the NIRD resources for project storage, that is, on NIRD TS (`/nird/projects`) and NIRD DL (`/nird/datalake`). The two resources have separate quota based on the project allocation.

You can see the quota and the current usage by running:

```console
$ dusage -p NSxxxxK
```

```{note}
Notice, that quotas are shown in TiB (tebibyte), and not TB (terabyte).
```

 
