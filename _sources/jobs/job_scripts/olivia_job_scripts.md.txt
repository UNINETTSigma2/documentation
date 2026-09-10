---
orphan: true
---

(job-scripts-on-olivia)=

# Job Scripts on Olivia

This page documents how to specify the queue system parameters for the
different job types on Olivia. See {ref}`job-types-olivia`
for information about the different job types on Olivia.  See also
{ref}`stage-in-stage-out` for how to automatically copy files from and
to NIRD.

(job_scripts_olivia_small)=

## Small

The default type of job on Olivia is the *small* job.
It is meant for CPU jobs that need less than 256 CPUs.

_Small_ jobs must specify account (`--account`), walltime limit
(`--time`) and how much memory is needed.  Usually, they will also
specify the number of tasks (i.e., processes) to run (`--ntasks` - the
default is 1), and they can also specify how many cpus each task
should get (`--cpus-per-task` - the default is 1).

Memory usage is specified with `--mem-per-cpu`, in _MiB_
(`--mem-per-cpu=3600M`) or _GiB_ (`--mem-per-cpu=4G`).

_Small_ jobs can only use one node on Olivia, so one cannot ask for
more than one node or more than 256 tasks or cpus.

If a job tries to use more (resident) memory on a compute node than it
requested, it will be killed.  Note that it is the _total_ memory
usage on each node that counts, not the usage per task or cpu.  So,
for instance, if your job has two single-cpu tasks and asks
for 2 GiB RAM per cpu, the total limit will be 4 GiB.
The queue system does not care if one of the tasks uses more than 2
GiB, as long as the total usage on the node is not more than 4 GiB.

A typical job specification for a small job would be

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --mem-per-cpu=2G
	#SBATCH --ntasks=128

This will start 128 tasks (processes), each one getting one cpu and 2
GiB RAM.

To run multithreaded applications, use `--cpus-per-task` to allocate
the right number of cpus to each task.  `--cpus-per-task` sets the
environment variable `$OMP_NUM_THREADS` so that OpenMP programs by
default will use the right number of threads.  (It is possible to
override this number by setting `$OMP_NUM_THREADS` in the job script.)
For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --mem-per-cpu=2G
	#SBATCH --ntasks=2 --cpus-per-task=64

This job will run 2 processes on the node, each process getting 64
cpus.

All _small_ jobs on Olivia are allocated the requested cpus and memory
exclusively, but share nodes with other jobs.  Also note that they are
bound to the cpu cores they are allocated.  However, the tasks and
threads are free to use all cores the job has access to on the node.

The [Olivia Sample MPI Job](olivia/olivia_sample_mpi_job.md) page has an example
of a _small_ MPI job.


(job_scripts_olivia_large)=

## Large

_Large_ jobs are meant for CPU jobs that need at least 256 CPUs.  The
jobs get whole nodes.

_Large_ jobs must specify account (`--account`), walltime limit
(`--time`) and how many nodes are needed.  Usually, they will also
specify the number of tasks (i.e., processes) to run per node
(`--ntasks-per-node` - the default is 1), and they can also specify
how many cpus each task should get (`--cpus-per-task` - the default is
1).

_Large_ jobs gets one or more whole nodes, and are allocated and
accounted for the whole node, regardless of how may tasks they run on
each node.  The jobs get access to all of the memory on the node, and
one should not and cannot specify memory when submitting the jobs.

If a job tries to use more (resident) memory on a compute node exists
on the node, it will be killed.

A typical job specification for a large job would be

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=4
	#SBATCH --ntasks-per-node=256

This will start 256 tasks (processes) on each of 4 nodes.

To run multithreaded applications, use `--cpus-per-task` to allocate
the right number of cpus to each task.  `--cpus-per-task` sets the
environment variable `$OMP_NUM_THREADS` so that OpenMP programs by
default will use the right number of threads.  (It is possible to
override this number by setting `$OMP_NUM_THREADS` in the job script.)
For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=2
	#SBATCH --ntasks-per-node=4 --cpus-per-task=64

This job will run 4 processes on each of the 2 nodes, each process
getting 64 cpus.


(job_scripts_olivia_accel)=

## Accel
_Accel_ jobs are specified just like *small* jobs except that they
also have to specify `--partition=accel`.  In addition, they must also
specify how many GPUs to use, and how they should be distributed
across nodes and tasks.  The simplest way to do that is, with
`--gpus=N` or `--gpus-per-node=N`, where `N` is the number of GPUs to
use.

For a job simple job running one process and using one GPU, the
following example is enough:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=accel --gpus=1
    #SBATCH --time=1-0:0:0
    #SBATCH --mem-per-cpu=8G

Here is an example that asks for 2 tasks and 2 GPUs on one node:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=accel
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks-per-node=2 --gpus-per-node=2 --nodes=1
    #SBATCH --mem-per-cpu=8G

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

See [sbatch](https://slurm.schedmd.com/sbatch.html) or `man sbatch`
for the details, and other GPU related specifications.

(The old way of specifying GPUs: `--gres=gpu:N` is still supported,
but is less flexible than the above specification.)

The [Olivia Sample GPU Job](olivia/olivia_sample_gpu_job.md) page has an example
of a GPU job.


(job_scripts_olivia_devel)=

## Devel

_Devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a regular
job, except that it has restrictions on job length and size.

For instance:

    #SBATCH --account=MyProject
    #SBATCH --job-name=MyJob
    #SBATCH --partition=small
    #SBATCH --qos=devel
    #SBATCH --time=00:30:00
    #SBATCH --ntasks=16
