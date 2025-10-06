---
orphan: true
---

(job-scripts-on-olivia)=

# Job Scripts on Olivia

This page documents how to specify the queue system parameters for the
different job types on Olivia. See {ref}`job-types-olivia`
for information about the different job types on Olivia.

(job_scripts_olivia_normal)=

## Normal

The basic type of job on Olivia is the *normal* job.

_Normal_ jobs must specify account (`--account`), walltime limit
(`--time`) and how much memory is needed.  Usually, they will also
specify the number of tasks (i.e., processes) to run (`--ntasks` - the
default is 1), and they can also specify how many cpus each task
should get (`--cpus-per-task` - the default is 1).

The jobs can also specify how man tasks should be run per node
(`--ntasks-per-node`), or how many nodes the tasks should be
distributed over (`--nodes`).  Without any of these two
specifications, the tasks will be distributed on any available
resources.

Memory usage is specified with `--mem-per-cpu`, in _MiB_
(`--mem-per-cpu=3600M`) or _GiB_ (`--mem-per-cpu=4G`).

If a job tries to use more (resident) memory on a compute node than it
requested, it will be killed.  Note that it is the _total_ memory
usage on each node that counts, not the usage per task or cpu.  So,
for instance, if your job has two single-cpu tasks on a node and asks
for 2 GiB RAM per cpu, the total limit on that node will be 4 GiB.
The queue system does not care if one of the tasks uses more than 2
GiB, as long as the total usage on the node is not more than 4 GiB.

A typical job specification for a normal job would be

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --mem-per-cpu=2G
	#SBATCH --ntasks=128

This will start 128 tasks (processes), each one getting one cpu and 2
GiB RAM.  The tasks can be distributed on any number of nodes (up to
128, of course).

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
	#SBATCH --ntasks=8 --cpus-per-task=64 --ntasks-per-node=4

This job will get 2 nodes, and run 4 processes on each of them, each
process getting 64 cpus.  All in all, that will be two whole nodes on
Olivia.

All jobs on Olivia are allocated the requested cpus and memory
exclusively, but share nodes with other jobs.  Also note that they are
bound to the cpu cores they are allocated.  However, the tasks and
threads are free to use all cores the job has access to on the node.

Note that the more restrictive one is in specifying how tasks are
placed on nodes, the longer the job might have to wait in the job
queue: In this example, for instance, there might be eight nodes with
64 idle cpus, but not two whole idle nodes.  Without the
`--ntasks-per-node` specification, the job could have started now, but
with the specification, it will have to wait.

The [Olivia Sample MPI Job](olivia/olivia_sample_mpi_job.md) page has an example
of a _normal_ MPI job.


(job_scripts_olivia_accel)=

## Accel
_Accel_ jobs are specified just like *normal* jobs except that they
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

_Devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a _normal_
job, except that it has restrictions on job length and size.

For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --qos=devel
	#SBATCH --time=00:30:00
	#SBATCH --ntasks=16
