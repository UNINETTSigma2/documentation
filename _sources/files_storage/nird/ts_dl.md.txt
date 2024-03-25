# NIRD TS vs NIRD DL

NIRD consists of two separate storage systems, namely Tiered Storage (NIRD TS) and 
Data Lake (NIRD DL).

NIRD TS has several tiers spanned by single filesystem and designed for performance and used mainly for active project data.

NIRD DL has a flat structure, designed mainly for less active data, sharing data across multiple projects, and interfacing with external storages.

Both are based on IBM Elastic Storage System. 


## Architecture comparison

| |  NIRD TS  |  NIRD DL  |
| :------------- | :------------- | :------------- |
| Tiers | Performance and capacity tiers<br> Automatic, transparent tiering<br>Dedicated pools for metadata | Flat architecture (no tiers)  |
| Designed for | - active project data<br>- data processing<br>- AI workloads| - less active data<br>- data libraries<br>- sharing data across multiple projects<br>- interfacing with external storages |
| Data integrity secured by | - erasure coding<br> - snapshots <br> - backup[^1] | - erasure coding<br> - snapshots | 

## Functionality comparison

| |  NIRD TS  |  NIRD DL  |
| :------------- | :------------- | :------------- |
| Protocols| POSIX, GPFS and NFS | POSIX, GPFS and S3[^2] |
| APIs | GPFS, Discover REST API[^3] | GPFS, S3, Discover REST API[^3] |
| Possibilities for| - file access logs<br>-data insight: metadata harvesting[^3] | - file access logs<br>- data insight: metadata harvesting[^3]<br>- encrypted projects |
| Access controls | - ACLs<br>- extended attributes | - ACLs<br>- extended attributes<br>- RBAC via S3[^2] |

## Filesystems

### NIRD TS
- Project storage `/nird/projects`
- User’s home `/nird/home`
- Scratch storage `/nird/scratch`[^4]
- Archive `/archive`[^5]

### NIRD DL
- Project storage `/nird/datalake`
- Backup `/backup`[^5]
- Archive `/archive`[^5]



--
[^1]: optional, see [backup page](backup_lmd.md)
[^2]: to be enabled Q4 2023
[^3]: available at the moment only for internal purposes, plans for testing with pilot projects
[^4]: available on NIRD login nodes only
[^5]: not accessible to users