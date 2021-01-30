# Storage areas on HPC clusters

Projects and users receive different areas to store files and other
data. Some areas are used for temporary files during job execution
while others are for storing project data.

- [Overview](#overview)
- [Usage and quota](#usage-and-quota)
- [Home directory](#home-directory)
- [Job scratch area](#job-scratch-area)
- [Job scratch area on local disk](#job-scratch-area-on-local-disk)
- [User work area](#user-work-area)
- [Project area](#project-area)
- [Shared project area](#shared-project-area)

(clusters-overview)=

## Overview

The following table summarizes the different storage options for **Betzy, Fram and Saga**.
Below the table we give recommendations and discuss pros and cons for the various storage areas.

| Directory                                       | Purpose              | Default Quota                      | [Backup](backup.md)                 |
| :---------------------------------------------- | :------------------- | :--------------------------------- | :---------------------------------: |
| `/cluster/home/$USER` (`$HOME`)                 | User data            | 20 GiB / 100 K files               | [Only if quota enforced](backup.md) |
| `/cluster/work/jobs/$SLURM_JOB_ID` (`$SCRATCH`) | Per-job data         | N/A                                | No                                  |
| (Only Saga) `/localscratch/$SLURM_JOB_ID` (`$LOCALSCRATCH`) | Per-job data | [Individual](#job-scratch-area-on-local-disk) | No                   |
| `/cluster/work/users/$USER` (`$USERWORK`)       | Staging and job data | N/A                                | No                                  |
| `/cluster/projects/<project_name>`              | Project data         | [1 TiB / 1 M files](#project-area) | Yes                                 |
| `/cluster/shared/<folder_name>`                 | Shared data          | [Individual](#shared-project-area) | No                                  |

- **User areas and project areas are private**: Data handling and storage policy is documented [here](/files_storage/sharing_files.md).
- **Note that the `$LOCALSCRATCH` area is only implemented on Saga**
- In addition to the areas in the tables above, **both clusters mount the
  NIRD project areas** as `/nird/projects/nird/NSxxxxK` on the login nodes
  (but not on the compute nodes).
- The `/cluster` file system is a high-performance parallel file
  system.  On Fram, it is a [Lustre](https://www.lustre.org/) system with
  a total storage space of 2.3PB, and on Saga it is a
  [BeeGFS](https://www.beegfs.io/) system with a total storage space of
  1.1PB.
  For performance optimizations, consult
  [Lustre performance tips](/files_storage/performance/lustre.md) (Fram)
  and
  [BeeGFS performance tips](/files_storage/performance/beegfs.md) (Saga)
  pages.


## Usage and quota

You can see the **disk usage and disk quotas** of your available areas with:
```
$ dusage
```
But please note that this command does not show you the number of files and
there may be a quota on the number of files.

To see the usage **including the number of files**, check:
```
$ dusage -i
```

On **Saga** you can see a **project's disk usage**, e.g., for nn1234k, with:
```
$ beegfs-ctl --getquota --gid nn1234k
```

```{warning}
**Frequently asked questions**

- **I cannot copy files although we haven't used up all space**:
  You have probably exceeded the quota on the number of files.

- **I have moved files to the project folder but my home quota usage did not go down**:
  Moving files does not change ownership of the files. You need to also change the ownership of the files
  in the project folder from you to the group (change the ownership from 'username_g' to 'username').
  Please refer to the [example on changing file ownership](change_file_ownership.md).
```

(clusters-homedirectory)=

## Home directory

The home directory is `/cluster/home/$USER`. The location is stored
in the environment variable `$HOME`.  A quota is enabled on home
directories which is by default 20 GiB and 100 000 files, so it
is not advisable to run jobs in `$HOME`. However, it is perfectly
fine to store `stderr` and `stdout` logs from your batch jobs in
`$HOME` so they are available for reviewing in case of issues with it.

The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.

The home directory is only accessible for the user. Files that should be
accessible by other uses in a project must be placed in the project
area.

Backed up with daily snapshots **only if quota is enforced** for the last 7
days and weekly snapshots for the last 6 weeks
([documentation about backup](backup.md)).


## Job scratch area

Each job gets an area `/cluster/work/jobs/$SLURM_JOB_ID` that is
automatically created for the job, and automatically deleted when the
job finishes.  The location is stored in the environment variable
`$SCRATCH` available in the job.  `$SCRATCH` is only accessible by the
user running the job.

The area is meant as a temporary scratch area during job
execution.
**This area is not backed up** ([documentation about backup](backup.md)).

There are special commands (`savefile` and `cleanup`) one can use in
the job script to ensure that files are copied back to the submit
directory `$SLURM_SUBMIT_DIR` (where `sbatch` was run).

```{note}
**Pros of running jobs in the job scratch area**

- There is less risk of interference from other jobs because every job ID has
  its own scratch directory.
- Because the scratch directory is removed when the job finishes, the scripts
  do not need to clean up temporary files.
```

```{warning}
**Cons of running jobs in the job scratch area**

- Since the area is removed automatically, it can be hard to debug
  jobs that fail.
- One must use the special commands to copy files back in case the job
  script crashes before it has finished.
- If the main node of a job crashes (i.e., not the job script, but the
  node itself), the special commands might not be run, so files might
  be lost.
```


## Job scratch area on local disk

**This only exists on Saga.**

A job on **Saga** can request a scratch area on local disk on the node
it is running on.  This is done by specifying
`--gres=localscratch:<size>`, where *<size>* is the size of the requested
area, for instance `20G` for 20 GiB.  Most nodes on Saga have 300 GiB
disk that can be handed out to local scratch areas; a few of the
bigmem nodes have 7 TiB and the GPU nodes have 8 TiB.  If a job tries
to use more space on the area than it requested, it will get a "disk
quota exceeded" or "no space left on device" error (the exact message
depends on the program doing the writing).

Jobs that request this, get an area `/localscratch/$SLURM_JOB_ID`
that is automatically created for the job, and automatically deleted
when the job finishes.  The location is stored in the environment
variable `$LOCALSCRATCH` available in the job.  `$LOCALSCRATCH` is
only accessible by the user running the job.

Note that since this area is on *local disk* on the compute node, it
is probably not useful for jobs running on more than one node (the job
would get one independent area on each node).

The area is meant to be used as a temporary scratch area during job
execution by jobs who do a lot of disk IO operations (either metadata
operations or read/write operations).  Using it for such jobs will
speed up the jobs, and reduce the load on the `/cluster` file system.

**This area is not backed up** ([documentation about backup](backup.md)).

Currently, there are *no* special commands to ensure that files are
copied back automatically, so one has to do that with `cp` commands or
similar in the job script.

```{note}
**Pros of running jobs in the local disk job scratch area**

- IO operations are faster than on the `/cluster` file system.
- It reduces the load on the `/cluster` file system.
- There is less risk of interference from other jobs because every job ID has
  its own scratch directory.
- Because the scratch directory is removed when the job finishes, the scripts
  do not need to clean up temporary files.
```

```{warning}
**Cons of running jobs in the local disk job scratch area**

- Since the area is removed automatically, it can be hard to debug
  jobs that fail.
- One must make sure to use `cp` commands or similar in the job
  script to copy files back.
- If the main node of a job crashes (i.e., not the job script, but the
  node itself), files might be lost.
```


## User work area

Each user has an area `/cluster/work/users/$USER`.  The location is
stored in the environment variable `$USERWORK`.
**This area is not backed up** ([documentation about backup](backup.md)).
`$USERWORK` is only accessible by
the user owning the area.

This directory is meant for files that are used by one or more jobs.
All result files must be moved out from this area after the jobs
finish, otherwise they will be automatically deleted after a while
(see notes below). We highly encourage users to keep this area tidy,
since both high disk usage and automatic deletion process takes away
disk performance. The best solution is to clean up any unnecessary
data after each job.  Since `$USERWORK` is only accessible by the
user, you must move results to the project area if you want to share
them with other people in the project.

File deletion depends on the newest of the *creation-*, *modification-* and
*access* time and the total usage of the file system. The oldest files will
be deleted first and a weekly scan removes files older than 42 days.

When file system usage reaches 70%, files older than 21 days are subject to
automatic deletion.

It is **not** allowed to try to circumvent the automatic deletion by
for instance running scripts that touch all files.

```{note}
**Pros of running jobs in the user work area**

- Since job files are not removed automatically directly when a job
  finishes, it is easier to debug failing jobs.
- There is no need to use special commands to copy files back in case
  the job script or node crashes before the job has finished.
```

```{warning}
**Cons of running jobs in the user work area**

- There is a risk of interference from other jobs unless one makes
  sure to run each job in a separate sub directory inside `$USERWORK`.
- Because job files are not removed when the job finishes, one has to
  remember to clean up temporary files afterwards.
- One has to remember to move result files to the project area if one
  wants to keep them.  Otherwise they will eventually be deleted by
  the automatic file deletion.
```


(project-area)=

## Project area

All HPC projects have a dedicated local space to share data between project
members, located at **/cluster/projects/<project_name>**.

The project area is quota controlled and the default project quota for
HPC projects is 1 TB, but projects can apply for more during the
application process with a maximum quota of 10 TB.

Also after the project has been created, project members can request to increase
the quota to up to 10 TB by motivating why this is needed. Such requests should be submitted by the project leader via e-mail to <sigma2@uninett.no>.

Requests for more than 10 TB require an application for a separate [NIRD](nird.md) project area.

Note that unused quota can also be withdrawn for technical reasons (too little
space) or organisational reasons (less needs/less usage/less members of
group/less compute hrs).

Daily backup is taken to NIRD ([documentation about backup](backup.md)).

To see disk usage and quota information for your project, run `dusage -p <project_name>`.

```{note}
**Pros of running jobs in the project area**

- Since job files are not removed automatically directly when a job
  finishes, it is easier to debug failing jobs.
- There is no need to use special commands to copy files back in case
  the job script or node crashes before the job has finished.
- There is no need to move result files to save them permanently or
  give the rest of the project access to them.
```

```{warning}
**Cons of running jobs in the project area**

- There is a risk of interference from other jobs unless one makes
  sure to run each job in a separate sub directory inside the project
  area.
- Because job files are not removed when the job finishes, one has to
  remember to clean up temporary files afterwards, otherwise they can
  fill up the quota.
- There is a risk of using all of the disk quota if one runs many jobs
  and/or jobs needing a lot of storage at the same time.
```


## Shared project area

In special cases there might be a need for sharing data between projects for
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and it's according directory
in `/cluster/shared` is created.  The area gets a disk quota based on
the needs of the data.  The permissions of the areas vary.  In some
cases all but a few users in the meta-group only have read-only access
to the area.  In other cases, all users on the cluster have read-only
access.
