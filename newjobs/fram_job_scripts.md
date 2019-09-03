# Job Scripts on Fram

This page documents how to specify the queue system parameters for the
different job types on Fram.  See [Fram Job Types](fram_job_types.md)
for information about the different job types on Fram.

## Normal
The basic type of job on Fram is the *normal* job.

*Normal* jobs must specify account (`--account`), walltime limit
(`--time`) and number of nodes (`--nodes`).  Maximal wall time limit
for normal jobs is 7 days (168 hours).  By default, they must use
between 4 and 32 nodes, but a project can be allowed to use more nodes
if needed. The jobs can specify how many tasks should run per node and
how many CPUs should be used by each task.

If more than 32 nodes is needed, and the application in question can
actually scale more than 32 nodes, please send a request to
<support@metacenter.no>.

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
given number of cpus; it merely sets `$OMP_NUM_THREADS` so that OpenMP
jobs by default will use the right number of threads.  (It is possible
to override this number by setting `$OMP_NUM_THREADS` in the job
script.)

The [Fram Sample MPI Job](fram_sample_mpi_job.md) page has an example
of a normal MPI job.


## Preproc

## Bigmem

## Optimist

## Devel

## Short
