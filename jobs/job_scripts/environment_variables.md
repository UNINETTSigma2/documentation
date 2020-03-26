# Environment Variables in Job Scripts

Here is a list of some useful environment variables that can be used
in job scripts.  This is not a complete list.  See the [sbatch
documentation](https://slurm.schedmd.com/sbatch.html) for more
variables.  Another way to get a list of defined environment variables
is to run `env` in a job script and look at the output.

- `SLURM_JOB_ID`: The jobid returned by `sbatch`
- `SLURM_ARRAY_TASK_ID`: The id of the current array task in an [array
  job](array_jobs.md).
- `SLURM_JOB_NODELIST`: The list of nodes allocated to the job.
- `SLURM_NTASKS`: The number of tasks in the job.
- `SLURM_SUBMIT_DIR`: The directory where you ran `sbatch`.  Usually
  the place where the `slurm-<jobid>.out` is located.
- `SCRATCH`: A per-job scratch directory on the shared file system.
  See [work directory](work_directory.md) for details.
- `USERWORK`: A per-user scratch directory on the shared file system.
  See [work directory](work_directory.md) for details.
- `OMP_NUM_THREADS`: The number of threads to use for OpenMP
  programs.  This is controlled by the `--cpus-per-task` parameter to
  `sbatch`, and defaults to 1.
