(job_scripts_on_fram)=

# Job Scripts on Fram

This page documents how to specify the queue system parameters for the
different job types on Fram. See {ref}`job-types-fram`
for information about the different job types on Fram.


(job_scripts_fram_normal)=

## Normal

The basic type of job on Fram is the *normal* job.  Most of the other
job types are "variants" of a *normal* job.

*Normal* jobs must specify account (`--account`), walltime limit
(`--time`) and number of nodes (`--nodes`).  The jobs can specify how
many tasks should run per node and how many CPUs should be used by
each task.

A typical job specification for a normal job would be

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=10 --ntasks-per-node=32

This will start 32 tasks (processes) on each node, one for each cpu on the node.

All normal jobs gets exclusive access to whole nodes (all CPUs and
memory).  If a job tries to use more (resident) memory than is
configured on the nodes, it will be killed.  Currently, this limit is
60 GiB, *but it can change*.  If a job would require more memory per
task than the given 60 GiB split by 32 tasks, the trick is to limit the
number of tasks per node the following way:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=10 --ntasks-per-node=4

This example above will use only 4 tasks per node, giving each task 15
GiB.  Note that is the _total_ memory usage on each node that counts,
so one of the tasks can use more than 15 GiB, as long as the total is
less than 60 GiB.

If your job needs more than 60 GiB per task, the only option on Fram
is to use a *bigmem* job (see below).

To run multithreaded applications, use `--cpus-per-task` to allocate
the right number of cpus to each task.  For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=4 --ntasks-per-node=2 --cpus-per-task=16

Note that setting `--cpus-per-task` does *not* bind the tasks to the
given number of cpus for _normal_ jobs; it merely sets
`$OMP_NUM_THREADS` so that OpenMP jobs by default will use the right
number of threads.  (It is possible to override this number by setting
`$OMP_NUM_THREADS` in the job script.)

The [Fram Sample MPI Job](fram/fram_sample_mpi_job.md) page has an example
of a _normal_ MPI job.

See [Fram Job Placement](fram/fram_job_placement.md) for optional
parameters for controlling which nodes a _normal_ job is run on.


(job_scripts_fram_bigmem)=

## Bigmem

_Bigmem_ jobs must specify `--partition=bigmem`.  In addition, they
must specify wall time limit, the number of tasks and the amount of
memory memory per cpu.  A _bigmem_ job is assigned the requested cpus
and memory exclusively, but shares nodes with other jobs.  If a
_bigmem_ job tries to use more resident memory than requested, it gets
killed.  The maximal wall time limit for bigmem jobs is 14 days.

Here is an example that asks for 2 nodes, 3 tasks per node, 4 cpus per
task, and 32 GiB RAM per cpu:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --nodes=2 --ntasks-per-node=3 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

Note that even though the memory specification is called `--mem-per-cpu`, the
memory limit the job gets on the node is for the total usage by all processes
on the node, so in the above example, it would get a limit of 3 * 4 * 32 GiB =
384 GiB. The queue system doesn't care how the memory usage is divided between
the processes or threads, as long as the total usage on the node is below the
limit.

Also note that contrary to *normal* jobs, *bigmem* jobs _will_ be bound to the
cpu cores they are allocated, so the above sample job will have access to 12
cores on each node. However, the three tasks are free to use all cores the job
has access to on the node (12 in this example).

Here is a simpler example, which only asks for 16 tasks (of 1 cpu
each) and 32 GiB RAM per task; it does not care how the tasks are
allocated on the nodes:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks=16
    #SBATCH --mem-per-cpu=32G


(job_scripts_fram_devel)=

## Devel

_devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a _normal_
job, except that it has restrictions on job length and size.

For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
   	#SBATCH --qos=devel
	#SBATCH --time=00:30:00
	#SBATCH --nodes=2 --ntasks-per-node=32


(job_scripts_fram_short)=

## Short

_short_ jobs must specify `--qos=short`.  A _short_ job is like a _normal_
job, except that it has restrictions on job length and size.  It
differs from _devel_ jobs in that it allows somewhat longer and larger
jobs, but typically have longer wait time.

For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --qos=short
	#SBATCH --time=2:00:00
	#SBATCH --nodes=8 --ntasks-per-node=32


(job_scripts_fram_optimist)=

## Optimist

_Optimist_ jobs are specified just like _normal_ jobs, except that
they also must must specify `--qos=optimist`.  They run on the same
nodes as *normal* jobs.

An _optimist_ job can be scheduled if there are free resources at
least 30 minutes when the job is considered for scheduling.  However,
it can be requeued before 30 minutes have passed, so there is no
_gurarantee_ of a minimum run time.  When an _optimist_ job is requeued,
it is first sent a `SIGTERM` signal.  This can be trapped in order to
trigger a checkpoint.  After 30 seconds, the job receives a `SIGKILL`
signal, which cannot be trapped.

A simple _optimist_ job specification might be:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --partition=optimist
	#SBATCH --nodes=4 --ntasks-per-node=32
	#SBATCH --time=2:00:00
