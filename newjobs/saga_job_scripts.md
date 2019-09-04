# Job Scripts on Saga

This page documents how to specify the queue system parameters for the
different job types on Saga.  See [Saga Job Types](saga_job_types.md)
for information about the different job types on Saga.

## Normal

The basic type of job on Saga is the *normal* job.

_Normal_ jobs must specify account (`--account`), walltime limit
(`--time`) and how much memory is needed.  Maximal wall time limit for
normal jobs is 7 days (168 hours).  Usually, they will also specify
the number of tasks (i.e., processes) to run (`--ntasks` - the default
is 1), and they can also specify how many cpus each task should get
(`--cpus-per-task` - the default is 1).

The jobs can also specify how man tasks should be run per node
(`--ntasks-per-node`), or how many nodes the tasks should be
distributed over (`--nodes`).  Without any of these two
specifications, the tasks will be distributed on any available
resources.

Memory usage is specified with `--mem-per-cpu`, in _MiB_
(`--mem-per-cpu=3600M`) or _GiB_ (`--mem-per-cpu=4G`).  The _normal_
nodes on Saga have 186 GiB RAM and 40 cpus, which corresponds to 4.66
GiB per cpu.  If your program needs a lot of memory per cpu (more than
FIXME 8 GiB), then you should run it as a _bigmem_ job.

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
	#SBATCH --mem-per-cpu=3G
	#SBATCH --ntasks=16

This will start 16 tasks (processes), each one getting one cpu and 3
GiB RAM.  The tasks can be distributed on any number of nodes (up to
16, of course).

To run multithreaded applications, use `--cpus-per-task` to allocate
the right number of cpus to each task.  `--cpus-per-task` sets the
environment variable `$OMP_NUM_THREADS` so that OpenMP programs by
default will use the right number of threads.  (It is possible to
override this number by setting `$OMP_NUM_THREADS` in the job script.)
For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --time=1-0:0:0
	#SBATCH --mem-per-cpu=4G
	#SBATCH --ntasks=8 --cpus-per-task=10 --ntasks-per-node=4

This job will get 2 nodes, and run 4 processes on each of them, each
process getting 10 cpus.  All in all, that will be two whole nodes on
Saga.

A _normal_ job is allocated the requested cpus and memory exclusively,
but shares nodes with other jobs.  Also note that they are bound to
the cpu cores they are allocated.  However, the tasks and threads are
free to use all cores the job has access to on the node.

Note that the more restrictive one is in specifying how tasks are
placed on nodes, the longer the job might have to wait in the job
queue: In this example, for instance, there might be eight nodes with
10 idle cpus, but not two whole idle nodes.  Without the
`--ntasks-per-node` specification, the job could have started, but
with the specification, it will have to wait.

The [Saga Sample MPI Job](saga_sample_mpi_job.md) page has an example
of a _normal_ MPI job.

## Bigmem

Saga has 28 nodes with twice the amount of memory as the _normal_ node
(377 GiB RAM on 40 cpus) and 8 nodes with 3021 GiB RAM and 64 cpus.
_bigmem_ jobs run on these nodes, and are specified exactly like the
_normal_ jobs except that you also have to specify
`--partition=bigmem`.  The maximal wall time limit for bigmem jobs is
14 days.

Here is an example that asks for 2 tasks, 4 cpus per task, and 32 GiB
RAM per cpu:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks=2 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

## Optimist

_Optimist_ jobs are specified just like _normal_ jobs, except that
they also must must specify `--qos=optimist`, and should *not*
specify wall time limit.  They can on any node on Saga.
They get lower priority than other jobs, but can start as soon as
there are free resources.  However, when any other job needs its
resources, the _optimist_ job is stopped and put back on the job
queue.  Therefore, all _optimist_ jobs must use checkpointing, and
access to run _optimist_ jobs will only be given to projects that
demonstrate that they can use checkpointing.

An _optimist_ job will only be scheduled if there are free resources
at least 30 minutes when the job is considered for scheduling.
However, it can be requeued before 30 minutes have passed, so there is
no gurarantee of a minimum run time.  When an _optimist_ job is
requeued, it is first sent a `SIGTERM` signal.  This can be trapped in
order to trigger a checkpoint.  After 30 seconds, the job receives a
`SIGKILL` signal, which cannot be trapped.

To be able submit `optimist` jobs the project has to send a request to
Sigma2 to get an optimist quota.

A simple _optimist_ job specification might be:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --partition=optimist
	#SBATCH --mem-per-cpu=3G
	#SBATCH --ntasks=16

## Devel

FIXME: Update!

_devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a _normal_
job, except that it can use at most FIXME cpus, and there is a limit of
how many nodes _devel_ jobs can use at the same time (currently 4), each user is
allowed to run only 1 _devel_ job simultaneously.  _devel_ jobs also have a 
maximum walltime of 30 minutes. On the other hand, they get higher priority than
other jobs, in order to start as soon as possible.

If you have _temporary_ development needs that cannot be fulfilled by the _devel_
job type, please contact us at <support@metacenter.no>.

## Short

FIXME: Remove(?)

_short_ jobs must specify `--qos=short`.  A _short_ job is like a _normal_
job, except that it can use between 1 and 10 nodes, and there is a limit of
how many nodes _short_ jobs can use at the same time (currently 16), each user is
allowed to run 2 _short_ job simultaneously.  _short_ jobs also have a 
maximum walltime of 2 hours. On the other hand, they get slightly higher priority 
than other jobs, in order to start reasonably faster.
