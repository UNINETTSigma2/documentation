Job related environment variables
=================================

Here we list some environment variables that are defined when you run a
job script. These is not a complete list. Please consult the SLURM
documentation for a complete list.

Job number:

    SLURM_JOB_ID
    SLURM_ARRAY_TASK_ID  # relevant when you are using job arrays

List of nodes used in a job:

    SLURM_JOB_NODELIST

Scratch directories:

    SCRATCH

This is an automatically created scratch directory for each job (currently
`/cluster/work/jobs/$SLURM_JOB_ID`), that is automatically deleted when the
job has finished.

	USERWORK

A permanent work area (`/cluster/work/users/$USER`).  It is advisable to use a
sub directory of this unless you use `$SCRATCH`.

Submit directory (this is the directory where you ran `sbatch`):

    SLURM_SUBMIT_DIR
    SUBMITDIR (for backward compatibility)

Default number of threads:

    OMP_NUM_THREADS

This is automatically set to the value specified in `--cpus-per-task`, (or 1 if
`--cpus-per-task` is not specified).

Task count:

    SLURM_NTASKS

Note that this is only set if `--ntasks` or `--ntasks-per-node` has been
specified.  (If neither is specified, Slurm runs one task per allocated node,
but `SLURM_NTASKS` is not set.  In such cases, one can use
`SLURM_JOB_NUM_NODES` instead.)

