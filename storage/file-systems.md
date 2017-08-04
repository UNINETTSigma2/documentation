# File Systems on Fram

Projects and users receive different file systems to store files and other data.

## Project Area

This is the file system for shared space for each project. Project area is
mounted to **/nird/projects**.

The project area is quota controlled and the default project quota for NOTUR projects is
10TB. The total file system size for projects is 4.5PB.

Geo-replication is set up between Tromsø and Trondheim.
For backup, snapshots are taken with the following frequency:
* daily snapshots of the last 7 days
* weekly snapshots of the last 5 weeks

## Work Area

Fram has two areas for temporary data storage:
1. large, shared storage for the whole cluster at **/cluster/work**;
2. small, local scratch space, individual to each compute node, available under
**/node/scratch**.

#### /cluster/work

/cluster/work is a high-performance parallel - [Lustre](http://lustre.org) -
file system with a total storage space of 2.3PB.

Contains two subdirectories: *jobs* and *users*.
* `/cluster/work/jobs`
  - scratch space for each job
  - automatically created and deleted by the queue system
* `/cluster/work/users/<username>`
  - semi-permanent scratch space for each user set to the `$USERWORK` variable
  - it is subject to automatic deletion. See [Storage Policies](storage-policies.md).
  - it is not backed up. See [Backup](backup.md)

For performance optimizations, please consult [this](performance-tips.md) page.

#### /node/scratch

This storage space is fairly small and relies on local disk, thus being slow
and recommended only when local temporary space is needed. The scratch directory is set to the `$SCRATCH` variable.

## User area (home directories)

The user area is mounted to **/nird/home/<username>** and is set to the `$HOME` variable.

The total size is 0.5PB. This file system is small and is **not** suitable for running jobs. A quota is enabled on home directories which is by default 20GB per user.

The user area is geo-replicated between Tromsø and Trondheim. Additionally daily
snapshots are taken and kept for the last 365 days.
