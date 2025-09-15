(storage-areas)=

# Storage areas on HPC clusters

Projects and users receive different areas to store files and other
data. Some areas are used for temporary files during job execution
while others are for storing project data.

```{contents} Table of Contents
```


(clusters-overview)=

## Overview

The following table summarizes the different storage options for **Betzy, Fram, and Saga**.
Below the table, we give recommendations and discuss the pros and cons of the various storage areas.

| Directory                                       | Purpose              | {ref}`Default Quota <storage-quota>` | {ref}`Backup <storage-backup>` | Note | 
| :---------------------------------------------- | :------------------- | :--------------------------------- | :---------------------------------: | :----:|
| `/cluster/home/$USER` (`$HOME`)                 | User data            | 20 GiB / 100 K files               | Snapshots (Only if quota enforced)            ||
| `/cluster/work/jobs/$SLURM_JOB_ID` (`$SCRATCH`) | Per-job data         | N/A                                | No                                  ||
| (Fram/Saga) `/localscratch/$SLURM_JOB_ID` (`$LOCALSCRATCH`) | Per-job data | {ref}`Individual <job-scratch-area-on-local-disk>` | No                   |Only Fram SAGA|
| `/cluster/work/users/$USER` (`$USERWORK`)       | Staging and job data | N/A                                | No                                  ||
| `/cluster/projects/<project_name>`              | Project data         | {ref}`1 TiB / 1 M files <project-area>` | No                                 ||
| `/cluster/shared/<folder_name>`                 | Shared data          | {ref}`Individual <shared-project-area>` | No                                  ||
| `/nird/datapeak/NSxxxxK`                         | NIRD Data Peak (TS) projects | | | Login nodes only|
| `/nird/datalake/NSxxxxK`                         | NIRD Data Lake (DL) projects | | | Login nodes only|

- **User areas and project areas are private**: Data handling and storage policy is documented [here](/files_storage/sharing_files.md).
- **`$LOCALSCRATCH` area is only implemented on Fram and Saga**.
- Clusters mount the NIRD project areas as `/nird/datapeak/NSxxxxK` for NIRD Data Peak (DP) projects and `nird/datalake/NSxxxxK` for NIRD Data Lake (DL) projects **on the login nodes only ** (not on the compute nodes).
- The `/cluster` file system is a high-performance parallel file
  system.  On Fram, it is a [Lustre](https://www.lustre.org/) system with
  a total storage space of 2.3 PB, and on Saga it is a
  [BeeGFS](https://www.beegfs.io/) system with a total storage space of
  6.5 PB.
  For performance optimizations, consult {ref}`storage-performance`.


(clusters-homedirectory)=

## Home directory

The home directory is `/cluster/home/$USER`. The location is stored
in the environment variable `$HOME`.  {ref}`storage-quota` is enabled on home
directories which is by default 20 GiB and 100 000 files, so it
is not advisable to run jobs in `$HOME`. However, it is perfectly
fine to store `stderr` and `stdout` logs from your batch jobs in
`$HOME` so they are available for reviewing in case of issues with it.

The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.

The home directory is only accessible for the user. Files that should be
accessible by other uses in a project must be placed in the project
area.

{ref}`Backed up <storage-backup>`
with daily snapshots **only if {ref}`storage-quota` is enforced** for the last 7
days and weekly snapshots for the last 6 weeks.


## Job scratch area

Each job gets an area `/cluster/work/jobs/$SLURM_JOB_ID` that is
automatically created for the job, and automatically deleted when the
job finishes.  The location is stored in the environment variable
`$SCRATCH` available in the job.  `$SCRATCH` is only accessible by the
user running the job.

On Fram and Saga there are two scratch areas (see also below).

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

(job-scratch-area-on-local-disk)=

## Job scratch area on local disk

**This only exists on Fram and Saga**.

A job on **Fram/Saga** can request a scratch area on local disk on the node
it is running on.  This is done by specifying
`--gres=localscratch:<size>`, where *<size>* is the size of the requested
area, for instance `--gres=localscratch:20G` for 20 GiB.

Normal compute nodes on Fram have 198 GiB disk that can be handed out
to local scratch areas, and the bigmem nodes have 868 GiB.
On Saga most nodes have 330 GiB; a few of the
bigmem nodes have 7 TiB, the hugemem nodes have 13 TiB and the GPU
nodes have either 406 GiB or 8 TiB.  If a job tries
to use more space on the area than it requested, it will get a "disk
quota exceeded" or "no space left on device" error (the exact message
depends on the program doing the writing).
Please do not ask for more than what you actually need, other users might share
the local scratch space with you (Saga only).

Jobs that request a local scratch area, get an area `/localscratch/$SLURM_JOB_ID`
that is automatically created for the job, and automatically deleted
when the job finishes.  The location is stored in the environment
variable `$LOCALSCRATCH` available in the job.  `$LOCALSCRATCH` is
only accessible by the user running the job.

Note that since this area is on *local disk* on the compute node, it
is **probably not useful for jobs running on more than one node** (the job
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

- Input/output operations are faster than on the `/cluster` file system.
- Great if you need to write/read a large number of files.
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
- Not suitable for files larger than 198-300 GB.
- One must make sure to use `cp` commands or similar in the job
  script to copy files back.
- If the main node of a job crashes (i.e., not the job script, but the
  node itself), files might be lost.
```


(user-work-area)=

## User work area

Each user has an area `/cluster/work/users/$USER`.  The location is
stored in the environment variable `$USERWORK`.
**This area is not backed up** ([documentation about backup](backup.md)).
By default, `$USERWORK` is a private area and only accessible by
the user owning the area. However, it is possible to grant other
users access here, for e.g., debugging purposes. Note that write 
access to your `$USERWORK` can not be granted to others.

To allow others to read your work area, you may use the command: 
`chmod o+rx $USERWORK`
  
Note that by doing so you will allow everyone on the machine to 
access your user work directory. If you want to share the results
in `$USERWORK` with other people in the project, the best way is to 
move them to the project area. 

The `$USERWORK` directory is meant for files that are used by one 
or more jobs. All result files must be moved out from this area 
after the jobs finish, otherwise they will be automatically deleted
after a while (see notes below). We highly encourage users to keep 
this area tidy, since both high disk usage and automatic deletion
process takes away disk performance. The best solution is to clean up
any unnecessary data after each job.  

File deletion depends on the newest of the *creation-*, *modification-* and
*access* time and the total usage of the file system. The oldest files will
be deleted first and a weekly scan removes files older than 21 days
(up to 42 days if sufficient storage is available).

When file system usage reaches 70%, files older than 21 days are subject to
automatic deletion. If usage is over 90%, files older than 17 days are subject to
automatic deletion.

**It is not allowed to try to circumvent the automatic deletion by
for instance running scripts that touch all files.**

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
members, located at `/cluster/projects/<project_name>`.

The project area is controlled by {ref}`storage-quota` and the default project quota for
HPC projects is 1 TiB, but projects can apply for more during the
application process with a maximum quota of 10 TiB on Fram and Saga, and 20 TiB on Betzy.

Also after the project has been created, project members can request to increase
the quota to up to 10/20 TiB by documenting why this is needed. Such requests should be submitted by the project leader via e-mail to [contact@sigma2.no](mailto:contact@sigma2.no?subject=Storage%20Quota%20Request%20project%20X&body=1.%20How%20large%20are%20the%20input%20files%3F%20(Approximate%20or%20exact%20numbers%20are%20fine.)%0A%0A2.%20How%20many%20such%20input%20files%20will%20be%20used%20in%20a%20single%20job%3F%0A%0A3.%20At%20what%20rate%20do%20you%20intend%20to%20process%20your%20data%3F%20(Approximate%20GB%20per%20week%20or%20equivalent.)%0A%0A4.%20What%20size%20is%20your%20output%20files%20and%20will%20you%20use%20this%20data%20as%20input%20in%20further%20analysis%3F%0A%0A5.%20Please%20explain%20why%20you%20cannot%20benefit%20from%20the%20%2Fcluster%2Fwork%20area%0A%0A6.%20Based%20on%20your%20answers%20above%2C%20how%20much%20storage%20quota%20do%20you%20think%20you%20need%3F)
. Note that only files that are relevant for further computation jobs should be kept on the HPC machine. HPC is not intended for long-term storage. In your request, please include answers to the following questions:

1. How large are the input files? (Approximate or exact numbers are fine.)
2. How many such input files will be used in a single job?
3. At what rate do you intend to process your data? (Approximate GB per week or equivalent.)
4. What size are your output files and will you use this data as input in further analysis?
5. Please explain why you cannot benefit from the /cluster/work area
NIRD is tightly connected with our HPC systems and data can be moved between the two both fast and easily.
6. Please explain why staging data from NIRD is not sufficient for your project
7. Based on your answers above, how much storage quota do you think you need?


Requests for more than 10/20 TiB require an application for a separate {ref}`nird` project area. On special occasions, storage above 10/20 TiB can be permitted. This requires an investigation of the workflow to ensure that needs cannot be satisfied through an allocation on NIRD. Granted disk space above 10/20 TiB is charged according to the [Contribution model](https://www.sigma2.no/user-contribution-model), Storage category B.

Note that unused quota can also be withdrawn for technical reasons (too little
space) or organisational reasons (less needs/less usage/fewer members of
the group/fewer compute hours).

- **This area is not backed up**([documentation about backup](backup.md)).


```{note}
**Pros of running jobs in the project area**

- Since job files are not removed automatically when a job
  finishes, it is easier to debug failing jobs.
- There is no need to use special commands to copy files back in case
  the job script or node crashes before the job has finished.
- There is no need to move result files after the job finishes or
  give the rest of the project access to them.
```

```{warning}
**Cons of running jobs in the project area**

- There is a risk of interference from other jobs unless one makes
  sure to run each job in a separate sub-directory inside the project
  area.
- Because job files are not removed when the job finishes, one has to
  remember to clean up temporary files afterwards otherwise they can
  fill up the quota.
- There is a risk of using all of the disk quota if one runs many jobs
  and/or jobs needing a lot of storage at the same time.
- There is a risk in using project area for permanent storage, as it is not backed up. 
```

(shared-project-area)=

## Shared project area

In special cases, there might be a need for sharing data between projects for
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and its according directory
in `/cluster/shared` is created.  The area gets a disk quota based on
the needs of the data.  The permissions of the areas vary.  In some
cases all but a few users in the meta-group only have read-only access
to the area.  In other cases, all users on the cluster have read-only
access.

## Decommissioning

Starting at the 2020.1 resource allocation period, storage decommissioning
procedures have been established for the HPC storages. This to ensure a
predictable storage for users and projects, and the provisioning more
sustainable to Sigma2.
For more details, please visit the
[data decommissioning policies](https://www.sigma2.no/data-decommissioning-policies)
page.
