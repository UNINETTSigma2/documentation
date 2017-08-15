# Storage Systems on Fram

Projects and users receive different file systems to store files and other data.

## Project Area

This is the file system for shared space for each project. Project area is
mounted to **/nird/projects**.

The project area is quota controlled and the default project quota for NOTUR projects is
10TB. The total file system size for projects is 4.5PB.

Geo-replication is set up between Tromsø and Trondheim.
For backup, snapshots are taken with the following frequency:
* daily snapshots of the last 7 days and weekly snapshots of the last 5 weeks. See [Backup](backup.md).
* the total size of the project area is 4.5PB and the default size for Notur projects is 10TiB.


## Work Area

The **/cluster/work** is used as a work area for storing files during job execution.
/cluster/work is a high-performance parallel - [Lustre](http://lustre.org) -
file system with a total storage space of 2.3PB. The work area is not backed up. See [Backup](backup.md)

The work area contains two subdirectories: *jobs* and *users*.

 **1. `/cluster/work/jobs/$SLURM_JOB_ID`**

    This area is automatically created when a job starts and deleted when the
    job finishes. There are special commands (e.g., `savefile`) one can use in the job
    script to ensure that files are copied back to the submit directory (where you executed `sbatch`).

**Notes:**

* set to the `$SCRATCH` variable
* it is automatically deleted as soon as the job finishes.  Use `savefile` or `cleanup` in the batch script to
  ensure files are copied back before the directory is deleted.

**2. `/cluster/work/users/$USER`**
    This directory is meant for staging files that are used by one or more jobs.
    All data after processing must be moved out from this area or deleted after
    use, otherwise it will be automatically deleted after a while. We highly
    encourage users to keep this area tidy, since both high disk usage
    and automatic deletion process takes away disk performance. The best
    solution is to clean up any unnecessary data after each job.

**Notes:**

* set to the `$USERWORK` variable
* deletion depends on the newest of the *creation-*, *modification-* and *access*
  time and the total usage of the file system.
* the oldest files will be deleted first
* weekly scan removes files older than 42 days.
* when file system usage reaches 70%, files older than 21 days become subject for automatic deletion.
* it is not backed up. See [Backup](backup.md).

For performance optimizations, please consult [this](performance-tips.md) page.

## User area (home directories)

The user area is mounted to **/nird/home/<username>**. This file system is small and is **not** suitable for running jobs. A quota is enabled on home directories which is by default 20GB per user. * It is advisable to store `stderr` and `stdout` logs from your batch jobs in `$HOME` so they are available for reviewing in case of issues with it. The user area is geo-replicated between Tromsø and Trondheim. Additionally daily
snapshots are taken and kept for the last 365 days.

**Notes**
* set to the `$HOME` variable
* The total size of /nird/home is 0.5PB and a quota per user is set to 20GB.
* Users' home directory are meant to be personal and must **not** be shared
with anyone else, no matter its content.
* The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.
* backed up with daily snapshots for the last 365 days.See [Backup](backup.md).
