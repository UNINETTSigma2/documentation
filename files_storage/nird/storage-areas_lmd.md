# Storage areas, quota, and backup


## Home directories

Each user has a home directory `/nird/home/<username>`, where
`<username>` is the username. The default quota for home directories
is 20 GiB and 100 000 files. To check the disk usage and quotas, type:

```console
$ dusage
```

Home directories on NIRD also contain a backup of Betzy, Fram and Saga home
directories (when relevant) in `/nird/home/<username>/backup/fram` and
`/nird/home/<username>/backup/saga`.
To account for this default quota is doubled when relevant.
Note that this is a _backup_ from the HPC cluster; you cannot transfer
files to the cluster by putting them here.

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
You will keep the current quota on the old system until it is end of life.
You get the quota allocated by RFK for the 2022.2 allocation period on NIRD.
NIRD TS and NIRD DL will have separete quota based on the project allocation.
You can see the quota and the current usage by running:

```console
$ dusage -p NSxxxxK
```

## Software access

Software access is via command line. A number of softwares are already installed
for the users. If there is any particular software need, requests should be sent to 
support line.

 
