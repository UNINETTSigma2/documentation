# Job Work Directory

A job has multiple choices for its directory:

- Project area
- USERWORK
- SCRATCH

FIXME: list pros and cons, and give a recommendation.  Also mention
file copying, etc, and the alternative with signale before timeout.

We do _not_ recommend running jobs in your home directory, mainly
because the home directory quotas are small, so you risk your jobs
failing due to not being able to write to disk.  Also, the home
directories are private, so you would have to move the files to your
project area for others to be able to access them.
