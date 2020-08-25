# Job Work Directory

A job has multiple choices for its work directory, i.e., the directory
where it does its work:

- Project area (`/custer/projects/<projname>`)
- `$USERWORK` (`/cluster/work/users/$USER`)
- `$SCRATCH` (`/cluster/work/jobs/$SLURM_JOB_ID`)

There are different pros and cons with each of the choices.  See
[Storage Areas](../../files_storage/clusters.md) for details.

Currently, the recommended choice is to use the `$USERWORK` area.  It
provides a nice balance between auto-cleanup and simplicity.  Thus the
job script examples in this documentation will use `$USERWORK`.

We do _not_ recommend running jobs in your home directory, mainly
because the home directory quotas are small, so you risk your jobs
failing due to not being able to write to disk.  Also, the home
directories are private, so you would have to move the files to your
project area for others to be able to access them.

When using `$USERWORK`, it is a good idea to make sure that each job
runs in its own subdirectory.  This reduces the risk of jobs
interferring with each other.  One easy way to do that is to use the
following in the job script:

    ## Create and move to work dir
    workdir=$USERWORK/$SLURM_JOB_ID
	mkdir -p $workdir
	cd $workdir

Please remember to copy result files that you want to keep from
`$USERWORK` to your project area after the job has finished, because
files in `$USERWORK` are removed after a number of days.

If you are going to use `$SCRATCH`, there are two commands that can be
used in the job script to make sure result files are copied back even
if the job crashes before it finishes. (They don't give a 100%
guarantee: if the compute node itself crashes before the job finishes,
then the files will not be copied.)

	## Make sure file1, file2, etc are copied back to
	## $SLURM_SUBMIT_DIR at the end of the job:
    savefile <file1> <file2> ...

    ## Register a command to be run at the end of the job to copy
	## files somewhere
	cleanup <somecommand>

Both commands should be used in the job script _before_ starting the
main computation.  Also, if they contain any special characters like
`*`, they should be quoted.
