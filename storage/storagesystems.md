# Work Area and Storage on Fram

Projects and users receive different file systems to store files and other data. Some areas are used for temporary files during job execution while others are
for storing project data.

The following table summarizes the different temporary and project storage options. Read the following sections for specific details.

| Directory     | Purpose     | Default quota | Backup |
| :------------- | :------------- | :------------- | :------------- |
| `/cluster/work/jobs/$SLURM_JOB_ID`       | Job data. Set to `$SCRATCH`       | N/A**       | No       |
| `/cluster/work/users/$USER`      | Staging and cross-job data. Set to `$USERWORK`       | N/A**       | No       |
| `/nird/projects/fram/<project_name>`       | Project data       | 1TB       | Yes       |
| `/nird/home/<username>`       | User data. Set to `$HOME`       | 20GB       | Yes       |

** $SCRATCH contents are deleted after the job finishes execution and $USERWORK
content is deleted after a couple of weeks. See the policies about both in the sections below.

## Work Area

The **/cluster/work** is used as a temporary work area for storing files during job execution. The work area
is a high-performance parallel - [Lustre](http://lustre.org) -
file system with a total storage space of 2.3PB. The work area is not backed up. See [Backup](backup.md)

There are several reasons why the work areas are suitable for jobs:
* Work areas are on a faster file system than user home directories.
* There is less risk of interference from other jobs because every job ID has its own scratch directory
* Because the scratch directory is removed when the job finishes, the scripts do not need to clean up temporary files.
* It avoids taking unneeded backups of temporary and partial files, because $SCRATCH is not backed up.
More information about the automatic cleanup is found at the [Prolog and Epilog](../jobs/framqueuesystem.md##prolog_epilog).

The work area contains two subdirectories: *jobs* ($SCRATCH) and *users* ($USERWORK).

 #### 1. `/cluster/work/jobs/$SLURM_JOB_ID` ($SCRATCH) Directory

This area is automatically created when a job starts and deleted when the
job finishes. There are special commands (e.g., `savefile`) one can use in the job
script to ensure that files are copied back to the submit directory `$SLURM_SUBMIT_DIR` (where `sbatch` was run).

**Notes:**

* set to the `$SCRATCH` variable
* it is automatically deleted as soon as the job finishes.  Use `savefile` or `cleanup` in the batch script to
  ensure files are copied back before the directory is deleted.

#### 2. `/cluster/work/users/$USER` ($USERWORK) Directory

    This directory is meant for staging files that are used by one or more jobs.
    All data after processing must be moved out from this area or deleted after
    use, otherwise it will be automatically deleted after a while (see notes below). We highly
    encourage users to keep this area tidy, since both high disk usage
    and automatic deletion process takes away disk performance. The best
    solution is to clean up any unnecessary data after each job.

**Notes:**

* set to the `$USERWORK` variable
* file deletion depends on the newest of the *creation-*, *modification-* and *access* time and the total usage of the file system. The oldest files will be deleted first and a weekly scan removes files older than 42 days.
* when file system usage reaches 70%, files older than 21 days are subject to automatic deletion.
* it is not backed up. See [Backup](backup.md).

For performance optimizations, consult [Performance Tips](performance-tips.md) page.

## Project Area

This is the file system for shared space for each project. Project area is
mounted to **/nird/projects/fram/<project_name>**.

The project area is quota controlled and the default project quota for NOTUR projects is
1TB, but projects can apply for more during the application process or at a later
point in time if needed.

**Notes:**
* Geo-replication is set up between Tromsø and Trondheim.
* For backup, snapshots are taken with the following frequency:
    * daily snapshots of the last 7 days
    * weekly snapshots of the last 5 weeks. 
* See [Backup](backup.md).

## User Area ($HOME)

The user area is mounted to **/nird/home/<username>**. This file system is small
and is **not** suitable for running jobs. A quota is enabled on home directories
which is by default 20GB per user. It is advisable to store `stderr` and `stdout`
logs from your batch jobs in `$HOME` so they are available for reviewing in case
of issues with it. The user area is geo-replicated between Tromsø and Trondheim.
Additionally daily snapshots are taken and kept for the last 365 days.

**Notes**
* set to the `$HOME` variable
* The total size of /nird/home is 0.5PB and a quota per user is set to 20GB.
* Users' home directory are meant to be personal and must **not** be shared
with anyone else, no matter its content.
* The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.
* backed up with daily snapshots for the last 365 days.See [Backup](backup.md).
