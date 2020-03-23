# Storage areas on HPC clusters

Projects and users receive different areas to store files and other
data. Some areas are used for temporary files during job execution
while others are for storing project data.

- [Overview](#overview)
- [Usage and quota](#usage-and-quota)
- [Home directory](#home-directory)
- [Job scratch area](#job-scratch-area)
- [User work area](#user-work-area)
- [Project area](#project-area)
- [Shared project area](#shared-project-area)


## Overview

The following table summarizes the different storage options for **Betzy, Fram and Saga**.
Below the table we give recommendations and discuss pros and cons for the various storage areas.

| Directory                                       | Purpose              | Default Quota                      | [Backup](backup.md)                 |
| :---------------------------------------------- | :------------------- | :--------------------------------- | :---------------------------------: |
| `/cluster/home/$USER` (`$HOME`)                 | User data            | 20 GiB / 100 K files               | [Only if quota enforced](backup.md) |
| `/cluster/work/jobs/$SLURM_JOB_ID` (`$SCRATCH`) | Per-job data         | N/A                                | No                                  |
| `/cluster/work/users/$USER` (`$USERWORK`)       | Staging and job data | N/A                                | No                                  |
| `/cluster/projects/<project_name>`              | Project data         | [1 TiB / 1 M files](#project-area) | Yes                                 |
| `/cluster/shared/<folder_name>`                 | Shared data          | [Individual](#shared-project-area) | No                                  |

- **User areas and project areas are private**: Data handling and storage policy is documented [here](data_policy.md).
- In addition to the areas in the tables above, **both clusters mount the
  NIRD project areas** as `/nird/projects/nird/NSxxxxK` on the login nodes
  (but not on the compute nodes).
- The `/cluster` file system is a high-performance parallel file
  system.  On Fram, it is a [Lustre](http://lustre.org) system with
  a total storage space of 2.3PB, and on Saga it is a
  [BeeGFS](https://www.beegfs.io/) system with a total storage space of
  1.1PB.
  For performance optimizations, consult
  [Lustre performance tips](/storage/performance/lustre.md) (Fram)
  and
  [BeeGFS performance tips](/storage/performance/beegfs.md) (Saga)
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

To see the **project disk usage** (e.g. for nn1234k), use:
```
$ beegfs-ctl --getquota --gid nn1234k
```

<div class="alert alert-warning">
  <h4>Frequently asked questions</h4>
  <ul>
    <li>
      <b>I cannot copy files although we haven't used up all space</b>:
      You have probably exceeded the quota on the number of files.
    </li>
    <li>
      <b>I have moved files to the project folder but my home quota usage did not go down</b>:
      Moving files does not change ownership of the files. You need to also change the ownership of the files
      in the project folder from you to the group.
    </li>
  </ul>
</div>


## Home directory

The home directory is **/cluster/home/$USER**. The location is stored
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

Each job gets an area **/cluster/work/jobs/$SLURM_JOB_ID** that is
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

<div class="alert alert-success">
  <h4>Pros of running jobs in the job scratch area</h4>
  <ul>
    <li> There is less risk of interference from other jobs because every job ID has
         its own scratch directory.</li>
    <li> Because the scratch directory is removed when the job finishes, the scripts
         do not need to clean up temporary files.</li>
  </ul>
</div>

<div class="alert alert-danger">
  <h4>Cons of running jobs in the job scratch area</h4>
  <ul>
    <li> Since the area is removed automatically, it can be hard to debug
         jobs that fail.</li>
    <li> One must use the special commands to copy files back in case the job
         script crashes before it has finished.</li>
    <li> If the main node of a job crashes (i.e., not the job script, but the
         node itself), the special commands might not be run, so files might
         be lost.</li>
  </ul>
</div>


## User work area

Each user has an area **/cluster/work/users/$USER**.  The location is
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

<div class="alert alert-success">
  <h4>Pros of running jobs in the user work area</h4>
  <ul>
    <li> Since job files are not removed automatically directly when a job
         finishes, it is easier to debug failing jobs.</li>
    <li> There is no need to use special commands to copy files back in case
         the job script or node crashes before the job has finished.</li>
  </ul>
</div>

<div class="alert alert-danger">
  <h4>Cons of running jobs in the user work area</h4>
  <ul>
    <li> There is a risk of interference from other jobs unless one makes
         sure to run each job in a separate sub directory inside `$USERWORK`.</li>
    <li> Because job files are not removed when the job finishes, one has to
         remember to clean up temporary files afterwards.</li>
    <li> One has to remember to move result files to the project area if one
         wants to keep them.  Otherwise they will eventually be deleted by
         the automatic file deletion.</li>
  </ul>
</div>


## Project area

All HPC projects have a dedicated local space to share data between project
members, located at **/cluster/projects/<project_name>**.

The project area is quota controlled and the default project quota for
HPC projects is 1 TB, but projects can apply for more during the
application process or request at a later point in time if needed. The
maximum quota for the project area is 10TB. Greater needs will require
an application for a separate NIRD project area.

Daily backup is taken to NIRD ([documentation about backup](backup.md)).

To see disk usage and quota information for your project, run `dusage -p <project_name>`.

<div class="alert alert-success">
  <h4>Pros of running jobs in the project area</h4>
  <ul>
    <li>Since job files are not removed automatically directly when a job
        finishes, it is easier to debug failing jobs.</li>
    <li>There is no need to use special commands to copy files back in case
        the job script or node crashes before the job has finished.</li>
    <li>There is no need to move result files to save them permanently or
        give the rest of the project access to them.</li>
  </ul>
</div>

<div class="alert alert-danger">
  <h4>Cons of running jobs in the project area</h4>
  <ul>
    <li>There is a risk of interference from other jobs unless one makes
        sure to run each job in a separate sub directory inside the project
        area.</li>
    <li>Because job files are not removed when the job finishes, one has to
        remember to clean up temporary files afterwards, otherwise they can
        fill up the quota.</li>
    <li>There is a risk of using all of the disk quota if one runs many jobs
        and/or jobs needing a lot of storage at the same time.</li>
  </ul>
</div>


## Shared project area

In special cases there might be a need for sharing data between projects for
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and it's according directory
in `/cluster/shared` is created.  The area gets a disk quota based on
the needs of the data.  The permissions of the areas vary.  In some
cases all but a few users in the meta-group only have read-only access
to the area.  In other cases, all users on the cluster have read-only
access.
