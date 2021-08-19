(betzy_job_scripts)=

# Job Scripts on Betzy

## Accel
_Accel_ jobs are those that require GPUs to perform calculations. To ensure
that your job is run on only machinces with GPUs the `--partition=accel` flag
must be supplied. In addition, to get access to one or more GPUs one need to
request a number of GPUs with the `--gpus=N` flag (see below for more ways to
specify the number of GPUs for your job).

For a simple job, only requiring 1 GPU, the following example configuration
could be used:

```bash
#SBATCH --account=nn<XXXX>k
#SBATCH --job-name=SimpleGPUJob
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=accel
#SBATCH --gpus=1
```

The following example starts 2 tasks each with a single GPU. This is usefull
for MPI enabled jobs where each rank should be assigned a GPU.

```bash
#SBATCH --account=nn<XXXX>k
#SBATCH --job-name=MPIGPUJob
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=2
#SBATCH --partition=accel
#SBATCH --gpus-per-task=1
```

There are other GPU related specifications that can be used, and that
parallel some of the cpu related specifications.  The most useful are
probably:

- `--gpus-per-node` How many GPUs the job should have on each node.
- `--gpus-per-task` How many GPUs the job should have per task.
  Requires the use of `--ntasks` or `--gpus`.
- `--gpus-per-socket` How many GPUs the job should have on each
  socket.  Requires the use of `--sockets-per-node`.
- `--mem-per-gpu` How much RAM the job should have for each GPU.
  Can be used *instead of* `--mem-per-cpu`, (but cannot be used
  *together with* it).
