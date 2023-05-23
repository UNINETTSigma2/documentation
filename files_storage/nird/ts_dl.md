# NIRD Tiered Storage and Data Lake

NIRD consists of two separate storage systems, namely Tiered Storage (NIRD TS) and 
Data Lake (NIRD DL).

The NIRD TS have several tiers spanned by single filesystem and designed for active 
projects, while NIRD DL has a more flat structure, designed for less active data and 
sharing data across multiple projects and interfacing with external storages. 
Both are based on based on IBM Elastic Storage System. 

NIRD TS contains building blocks based on:
 
	- ESS 3200 building blocks for NVMe drives
	- ESS 5000 building blocks for NL-SAS drives

and the NIRD DL contains building blocks based only on:

	- ESS 5000 building blocks for NL-SAS drives

Below is the detail information of the difference between TS and DL.

## TS vs DL Capacity and Performance

|  Tiered Storage(TS)  |  Data Lake(DL)  |
| :------------- | :------------- |
| **Capacity :** 22 PB| **Capacity :** 11 PB |
| **Performance:**<br> Up to 400 Gb/s aggregated bandwidth<br> Aggregated I/O throughput ~ 209 GB/s | **Performance:**<br> Up to 200 Gb/s aggregated bandwidth<br> Aggregated I/O throughput ~ 66 GB/s |


## TS vs DL Architecture  

|  Tiered Storage(TS)  |  Data Lake(DL)  |
| :------------- | :------------- |
| Performance and capacity tiers<br> Automatic, transparent tiering | Flat architecture (no tiers)  |
| Designed for active projects | Designed for less active data, data libraries, sharing data across multiple projects and interfacing with external storages |
| Data integrity secured by:<br> - Erasure coding<br> - Snapshots <br> - Option for backup| Data integrity secured by: <br> - Erasure coding<br> - Snapshots | 

## TS vs DL Functionalities

|  Tiered Storage(TS)  |  Data Lake(DL)  |
| :------------- | :------------- |
| Protocols: POSIX, GPFS, NFS and SMB(Not yet enabled) | Protocols: POSIX, GPFS,S3(To be enabled later in 2023) |
| APIs: GPFS, Discover REST API(Initially to be used internally, might be offered for selected projects)       | APIs: GPFS, S3(To be enabled later in 2023) |
| Possibilities for:<br> - encrypted projects<br> - file access logs<br> -data insight:metadata harvesting (Initially to be used internally, might be offered for selected projects)| RBAC via S3 |
| ACLs and extended attributes | ACLs and extended attributes |

## TS vs DL Filesystems

|  Tiered Storage(TS)  |  Data Lake(DL)  |
| :------------- | :------------- |
| Project storage `/nird/projects`  | Data Lake project storage ` /nird/datalake` |
| User’s home `/nird/home`  | Backup for TS |
| Primary site for the NIRD Research Data Archive | Secondary site for the NIRD Research |
| Scratch storage `/nird/scratch` (Available only on NIRD login nodes) |  | 
 

NIRD TS(Resource: projects) and NIRD DL (Resource: datalake) will have separate quota based on the project allocation. You can see the quota and the current usage by running:

```console
$ dusage -p NSxxxxK
```

where `NSxxxxK` is the ID of the project.
