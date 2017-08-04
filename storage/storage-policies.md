# Storage Policies

This page lists the storage policies enforced by UNINETT Sigma2 on the HPC machines and NIRD.

## File System Structure

Projects have storage access for various uses. For more information about specifications, visit
the [File Systems](file-systems.md) page.

### /nird/home

* Users' home directory are meant to be personal and must **not** be shared
with anyone else, no matter its content.
* The home directory should be used for storing tools, scripts, application
sources or other relevant data which must have a backup.
* It is advisable to store `stderr` and `stdout` logs from your batch jobs in
	`$HOME` so they are available for reviewing in case of issues with it.

### /nird/projects

* The project areas can be used to share files within the projects.

### /cluster/work

* Temporary scratch space.
* `/cluster/work/users/$USER` is meant for staging files that are used by one
  or more jobs.  All data after processing must be moved out from this area or
  deleted after use, otherwise it will be automatically deleted after a while.
  We highly encourage users to keep this area tidy, since both high disk usage
  and automatic deletion process takes away disk performance. The best
  solution is to clean up any unnecessary data after each job.
* Alternatively, one can use `/cluster/work/jobs/$SLURM_JOB_ID` (a.k.a
  `$SCRATCH`) for jobs. This area is automatically created when a job starts,
  and deleted when the job finishes, so one does not have to clean up after
  the job. There are special commands (e.g., `savefile`) one can use in job
	script to ensure that files are copied back to the submit directory (where
	you executed `sbatch`).

## Automatic clean-up

* The user scratch area (`/cluster/work/users/`) is subject for automatic
deletion:
  - Deletion depends on newest of the creation-, modification- and access
	time and the total usage of the file system.
  - The oldest files will be deleted first.
  - Weekly scan removes files older than 42 days.
  - When file system usage reaches 70%, files older than 21 days become
  subject for automatic deletion.
* The job scratch area (`/cluster/work/jobs/$SLURM_JOB_ID`) is automatically
  deleted as soon as the job has finished.  Use `savefile` or `cleanup` to
  ensure files are copied back before the directory is deleted.

## Quotas

Default disk usage quotas are:
* `/nird/home/` - 20 GiB
* `/nird/projects/`	- 2 TiB
