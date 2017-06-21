# POLICIES

## Structure

* User's home directories are meant to be personal and must **not** be shared with anyone else, no matter of it's the content.
* Temporary, scratch space for each user MUST go under /cluster/work/user/$USER.
* Files between groups/projects can be shared in the project area.

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
