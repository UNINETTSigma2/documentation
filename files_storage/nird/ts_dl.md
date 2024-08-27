# NIRD Data Peak vs NIRD Data Lake

NIRD consists of two distinct storage systems, namely NIRD Data Peak (known internally as TS) and NIRD Data Lake (codenamed DL).

NIRD Data Peak has several tiers spanned by single filesystem and designed for performance and used mainly for active project data.

NIRD Data Lake has a flat structure, designed mainly for less active data, sharing data across multiple projects, and interfacing with external storages.

Both are based on IBM Elastic Storage System. 


## Architecture comparison

| |  NIRD Data Peak  |  NIRD Data Lake  |
| :------------- | :------------- | :------------- |
| Tiers | Performance and capacity tiers<br> Automatic, transparent tiering<br>Dedicated pools for metadata | Flat architecture (no tiers)  |
| Designed for | - high-performance storage for any type of active research data<br>- high-performance storage for I/O intensive computing such as HPC and AI workloads<br>- storage of valuable research data requiring a secondary copy<br>- storage for structured data of any volume larger than 1 TiB| - long-term storage of non-persistent data<br>- storage for any type of inactive/cold data<br>- storage for structured or unstructured data of any volume larger than 1 TiB<br>- sharing datasets and libraries for collaboration across projects and institutions in the sector<br>- interfacing with sensors or third party storage system<br>- object storage |
| Data integrity secured by | - erasure coding<br> - snapshots <br> - backup[^1] | - erasure coding<br> - snapshots | 

## Functionality comparison

| |  NIRD Data Peak  |  NIRD Data Lake  |
| :------------- | :------------- | :------------- |
| Protocols| POSIX, GPFS and NFS | POSIX, GPFS and S3[^2] |
| APIs | GPFS, Discover REST API[^3] | GPFS, S3, Discover REST API[^3] |
| Possibilities for| - file access logs<br>-data insight: metadata harvesting[^3] | - file access logs<br>- data insight: metadata harvesting[^3]<br>- encrypted projects |
| Access controls | - ACLs<br>- extended attributes | - ACLs<br>- extended attributes<br>- RBAC via S3[^2] |
| On-demand backup | Yes | No |

## Filesystems

### NIRD Data Peak
- Project storage `/nird/datapeak`
- User’s home `/nird/home`
- Scratch storage `/nird/scratch`[^4]
- Archive `/archive`[^5]

### NIRD DL
- Project storage `/nird/datalake`
- Backup `/backup`[^5]
- Archive `/archive`[^5]



--
[^1]: optional, see [backup page](backup_lmd.md)
[^2]: to be enabled Q2 2024
[^3]: available at the moment only for internal purposes, plans for testing with pilot projects
[^4]: available on NIRD login nodes only
[^5]: not accessible to users
