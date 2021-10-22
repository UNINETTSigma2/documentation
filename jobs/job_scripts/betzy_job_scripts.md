---
orphan: true
---

(job-scripts-on-betzy)=

# Job Scripts on Betzy

This page documents how to specify the queue system parameters for the
different job types on Betzy. See {ref}`job-types-betzy`
for information about the different job types on Betzy.

(job_scripts_betzy_normal)=

## Normal

The basic type of job on Betzy is the *normal* job.  Most of the other
job types are "variants" of a *normal* job.

*Normal* jobs must specify account (`--account`), walltime limit
(`--time`) and number of nodes (`--nodes`).  The jobs can specify how
many tasks should run per node and how many CPUs should be used by
each task.

A typical job specification for a normal job would be

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=10 --ntasks-per-node=128

This will start 128 tasks (processes) on each node, one for each cpu on the node.

All normal jobs gets exclusive access to whole nodes (all CPUs and
memory).  If a job tries to use more (resident) memory than is
configured on the nodes, it will be killed.  Currently, this limit is
245 GiB, *but it can change*.  If a job would require more memory per
task than the given 245 GiB split by 128 tasks, the trick is to limit the
number of tasks per node the following way:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=10 --ntasks-per-node=16

This example above will use only 16 tasks per node, giving each task 15
GiB.  Note that is the _total_ memory usage on each node that counts,
so one of the tasks can use more than 15 GiB, as long as the total is
less than 245 GiB.

To run multithreaded applications, use `--cpus-per-task` to allocate
the right number of cpus to each task.  For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=32

Note that setting `--cpus-per-task` does *not* bind the tasks to the
given number of cpus for _normal_ jobs; it merely sets
`$OMP_NUM_THREADS` so that OpenMP jobs by default will use the right
number of threads.  (It is possible to override this number by setting
`$OMP_NUM_THREADS` in the job script.)

The [Betzy Sample MPI Job](betzy/betzy_sample_mpi_job.md) page has an example
of a _normal_ MPI job.

(job_scripts_betzy_accel)=
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

The following example starts 2 tasks each with a single GPU. This is useful
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

(job_scripts_betzy_preproc)=

## Preproc

_Preproc_ jobs must specify `--partition=preproc`.  In addition, they
must specify wall time limit, the number of tasks and the amount of
memory memory per cpu.  A _preproc_ job is assigned the requested cpus
and memory exclusively, but shares nodes with other jobs.  (Currently,
there is only one node in the preproc partition.)  If a
_preproc_ job tries to use more resident memory than requested, it gets
killed.  The maximal wall time limit for preproc jobs is 1 day.

Here is an example that asks for 3 tasks per, 4 cpus per
task, and 2 GiB RAM per cpu:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=preproc
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks=3 --cpus-per-task=4
    #SBATCH --mem-per-cpu=2G

Note that even though the memory specification is called `--mem-per-cpu`, the
memory limit the job gets on the node is for the total usage by all processes
on the node, so in the above example, it would get a limit of 3 * 4 * 2 GiB =
12 GiB. The queue system doesn't care how the memory usage is divided between
the processes or threads, as long as the total usage on the node is below the
limit.

Also note that contrary to *normal* jobs, *preproc* jobs _will_ be
bound to the cpu cores they are allocated, so the above sample job
will have access to 12 cores. However, the three tasks are free to use
all cores the job has access to (12 in this example).


(job_scripts_betzy_devel)=

## Devel

_devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a _normal_
job, except that it has restrictions on job length and size.

For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
   	#SBATCH --qos=devel
	#SBATCH --time=00:30:00
	#SBATCH --nodes=2 --ntasks-per-node=128
>>>>>>> 423b84b4cdb41242d5ac5e5f4905de3158ff8dec
