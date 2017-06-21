# Available file systems on Fram

## User area (home directories)

Home directories are mounted to **/nird/home**. Total size is 0.5PB.

This file system is small and not suitable for running jobs. It should **NOT**
be used for running jobs.

Quota is enabled on home directories, default quota per user being 20GB.

User area is geo-replicated between Tromsø and Trondheim. Additionally daily
snapshots are taken and kept for the last 365 days.


## Project area

This is the file system for shared space for each project. Project area is 
mounted to **/nird/projects**. Total file system size for projects is 4.5PB.

Project area is quota controlled. Default project quota for NOTUR projects is 
10TB.

Geo-replication is set up between Tromsø and Trondheim.
For backup, snapshots are taken with the following frequency:
* daily snapshots for the last 7 days
* weekly snapshots for the last 5 weeks

## Work area 

Fram has two areas for temporary data storage:
* large, shared storage for the whole cluster, available under 
**/cluster/work**;
* small, local scratch space, individual to each compute node, available under
**/node/scratch**.

### /cluster/work

/cluster/work is a high-performance parallel - [Lustre](http://lustre.org) -
file system with a total storage space of 2.3PB.

Contains two subdirectories: *jobs* and *users*.
* /cluster/work/jobs
  - scratch space for each job
  - automatically created and deleted by the queue system
* /cluster/work/users
  - semi-permanent scratch space for each user
  - it is subject to [**automatic deletion**](storage-policies.md)
  - has [**no backup**](backup.md)

For performance optimizations, please consult [this](performance-tips.md) page.

### /node/scratch

This storage space is fairly small and relies on local disk, thus being slow
and recommended only when local temporary space is needed.
