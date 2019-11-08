# Storage areas on HPC clusters 

Projects and users receive different areas to store files and other
data. Some areas are used for temporary files during job execution
while others are for storing project data.

Data handling and storage policy is documented [here](data_policy.md).

The following tables summarizes the different storage options for the
two clusters.  Read the following sections for specific details.

* Fram

| Directory                                       | Purpose              | Default Quota                       | Backup |
| :-------------                                  | :-------------       | :----------:                        | :---:  |
| `/cluster/home/$USER` (`$HOME`)                 | User data            | 20 GiB / 100 K files                | Yes    |
| `/cluster/work/jobs/$SLURM_JOB_ID` (`$SCRATCH`) | Per-job data         | N/A                                 | No     |
| `/cluster/work/users/$USER` (`$USERWORK`)       | Staging and job data | N/A                                 | No     |
| `/cluster/projects/<project_name>`              | Project data         | 1 TiB[*](#project-area) / 1 M files | Yes    |
| `/cluster/shared/<folder_name>`                 | Shared data          | Individual[**](#shared-area)        | No     |

* Saga

| Directory                                       | Purpose              | Default Quota                       | Backup |
| :-------------                                  | :-------------       | :----------:                        | :---:  |
| `/cluster/home/$USER` (`$HOME`)                 | User data            | 20 GiB / 100 K files                | Soon   |
| `/cluster/work/jobs/$SLURM_JOB_ID` (`$SCRATCH`) | Per-job data         | N/A                                 | No     |
| `/cluster/work/users/$USER` (`$USERWORK`)       | Staging and job data | N/A                                 | No     |
| `/cluster/projects/<project_name>`              | Project data         | 1 TiB[*](#project-area) / 1 M files | Soon   |
| `/cluster/shared/<folder_name>`                 | Shared data          | Individual[**](#shared-area)        | No     |

Note that on __Saga__, backup of `$HOME` and
`/cluster/projects/<project_name>` has not been set up yet, but will
soon be implemented.

In addition to the areas in the tables above, both clusters mount the
NIRD project areas as `/nird/projects/nird/NSxxxxK` on the login nodes
(but not on the compute nodes).  (This has not been implemented on
Saga yet, but will happen soon.)

The **/cluster** file system is a high-performance parallel file
system.  On __Fram__, it is a [Lustre](http://lustre.org) system with
a total storage space of 2.3PB, and on __Saga__ it is a
[BeeGFS](https://www.beegfs.io/) system with a total storage space of
1.1PB.

For performance optimizations, consult [Performance Tips](performance_tips.md) page.

## Home Directory (`$HOME`)

The home directory is **/cluster/home/$USER**.  The location is stored
in the environment variable `$HOME`.  A quota is enabled on home
directories which is by default 20GiB and 100,000 files, so it
is not advisable to run jobs in `$HOME`. However, it is perfectly
fine to store `stderr` and `stdout` logs from your batch jobs in
`$HOME` so they are available for reviewing in case of issues with it.

`$HOME` is only accessible for the user.  Files that should be
accessible by other uses in a project must be placed in the project
area.

**Notes**

* The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.
* **Not implemented on Saga yet** Backed up with daily snapshots for
the last 7 days and weekly snapshots for the last 6 weeks. See
[Backup](backup.md).

## Job Scratch Area

Each job gets an area **/cluster/work/jobs/$SLURM_JOB_ID** that is
automatically created for the job, and automatically deleted when the
job finishes.  The location is stored in the environment variable
`$SCRATCH` available in the job.  `$SCRATCH` is only accessible by the
user running the job.

The area is meant as a temporary scratch area during job
execution. The area is _not_ backed up. See [Backup](backup.md).

There are special commands (`savefile` and `cleanup`) one can use in
the job script to ensure that files are copied back to the submit
directory `$SLURM_SUBMIT_DIR` (where `sbatch` was run).

### Pros of Running Jobs in `$SCRATCH`

* There is less risk of interference from other jobs because every job ID has
  its own scratch directory
* Because the scratch directory is removed when the job finishes, the scripts
  do not need to clean up temporary files.

### Cons of Running Jobs in `$SCRATCH`

* Since the area is removed automatically, it can be hard to debug
  jobs that fail.
* One must use the special commands to copy files back in case the job
  script crashes before it has finished.
* If the main node of a job crashes (i.e., not the job script, but the
  node itself), the special commands might not be run, so files might
  be lost.

## User Work Area

Each user has an area **/cluster/work/users/$USER**.  The location is
stored in the environment variable `$USERWORK`.  The area is _not_
backed up. See [Backup](backup.md).  `$USERWORK` is only accessible by
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

**Notes:**

* File deletion depends on the newest of the *creation-*, *modification-* and
  *access* time and the total usage of the file system. The oldest files will
  be deleted first and a weekly scan removes files older than 42 days.
* When file system usage reaches 70%, files older than 21 days are subject to
  automatic deletion.
* It is **not** allowed to try to circumvent the automatic deletion by
  for instance running scripts that touch all files.

### Pros of Running Jobs in `$USERWORK`

* Since job files are not removed automatically directly when a job
  finishes, it is easier to debug failing jobs.
* There is no need to use special commands to copy files back in case
  the job script or node crashes before the job has finished.

### Cons of Running Jobs in `$USERWORK`

* There is a risk of interference from other jobs unless one makes
  sure to run each job in a separate sub directory inside `$USERWORK`.
* Because job files are not removed when the job finishes, one has to
  remember to clean up temporary files afterwards.
* One has to remember to move result files to the project area if one
  wants to keep them.  Otherwise they will eventually be deleted by
  the atuomatic file deletion.

## <a name="project-area"></a>Project Area

All HPC projects have a dedicated local space to share data between project
members, located at **/cluster/projects/<project_name>**.

The project area is quota controlled and the default project quota for
HPC projects is 1TB, but projects can apply for more during the
application process or request at a later point in time if needed. The
maximum quota for the project area is 10TB. Greater needs will require
an application for a separate NIRD project area.

**Notes:**

* Daily backup is taken to NIRD. **Not implemented on Saga yet!**
* For backup, snapshots are taken with the following frequency:
    * daily snapshots of the last 7 days
    * weekly snapshots of the last 6 weeks. 
* To see disk usage and quota information for your project, run `dusage -p <project_name>`.
* See [Backup](backup.md).

## <a name="shared-area"></a>Shared Project Areas

In special cases there might be a need for sharing data between projects for 
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and it's according directory
in `/cluster/shared` is created.  The area gets a disk quota based on
the needs of the data.  The permissions of the areas vary.  In some
cases all but a few users in the meta-group only have read-only access
to the area.  In other cases, all users on the cluster have read-only
access.
