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

```{warning}
Please note that back up from HPC home directories to the new NIRD home 
is not yet enabled. This is expected to be activated close to the production phase.
```


## Scratch directories

The total storage space of `/scratch` is 15 TB.
Each user has a scratch directory `/scratch/<username>`. 
The area is meant as a temporary scratch area. This area is not backed up.
When file system usage reaches 75%, files are subject to automatic deletion.
There is no quota in the scratch area. 


## Project area

Each NIRD Data Storage project gets a project area `/nird/projects/NSxxxxK`,
where `NSxxxxK` is the ID of the project.

The project area has a quota on disk space and the number of files,
and you can see the quota and the current usage by running:
```console
$ dusage -p NSxxxxK
```

## Backup

In addition to geo-replication NIRD supports snapshots of project areas
and home directories allowing for recovery of deleted data.
For more information, visit the page {ref}`storage-backup`.

## Decommissioning

Starting at the 2020.1 resource allocation period, storage decommissioning
 procedures have been established for the NIRD project storage, to make 
 storage more predictable for the projects and the provisioning more 
 sustainable to Sigma2.
 For more details, please visit the 
[data decommissioning policies](https://www.sigma2.no/data-decommissioning-policies)
 page.
