# POLICIES

## Structure

### /nird/home

* User's home directories are meant to be personal and must **not** be shared with anyone else, no matter of it's the content.
* Home should be used for storing tools, scripts, application sources or other relevant data which must have a backup.

### /nird/projects

* Files between groups/projects can be shared in the project area.

### /cluster/work

* Temporary, scratch space for each user.
* /cluster/work/user/$USER should be used for running jobs, as a main storage during data processing. All data after processing must be moved out from this area or deleted after use.
* We highly encourage users to keep this area tidy, since both high disk
	usage and automatic deletion process takes away disk performance. The best
	solution is to clean up any unnecessary data after each job.

## Automatic clean-up

* Scratch area (/cluster/work) is subject for automatic deletion:
  - Deletion depends on newest of the creation-, modification- and access time and the total usage of the file system.
  - The oldest files will be deleted first.
  - Weekly scan removes files older than 42 days.
  - When file system usage reaches 70%, files older than 21 days become subject for automatic deletion.

## Quotas

Default quotas are the following:
* /nird/home			- 20GB
* /nird/projects	- 10TB
