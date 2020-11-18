(job_scripts_on_saga)=

# Job Scripts on Saga

This page documents how to specify the queue system parameters for the
different job types on Saga. See {ref}`job-types-saga`
for information about the different job types on Saga.

```{warning}
**On Saga use srun, not mpirun**

mpirun can get the number of tasks wrong and also lead to wrong task
placement. We don't fully understand why this happens. When using srun
instead of mpirun or mpiexec, we observe correct task placement on Saga.
```


(job_scripts_saga_normal)=

## Normal

The basic type of job on Saga is the *normal* job.

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

All jobs on Saga are allocated the requested cpus and memory
exclusively, but share nodes with other jobs.  Also note that they are
bound to the cpu cores they are allocated.  However, the tasks and
threads are free to use all cores the job has access to on the node.

Note that the more restrictive one is in specifying how tasks are
placed on nodes, the longer the job might have to wait in the job
queue: In this example, for instance, there might be eight nodes with
10 idle cpus, but not two whole idle nodes.  Without the
`--ntasks-per-node` specification, the job could have started, but
with the specification, it will have to wait.

The [Saga Sample MPI Job](saga/saga_sample_mpi_job.md) page has an example
of a _normal_ MPI job.


(job_scripts_saga_bigmem)=

## Bigmem

_Bigmem_ jobs are specified exactly like the _normal_ jobs except that
you also have to specify `--partition=bigmem`.

Here is an example that asks for 2 tasks, 4 cpus per task, and 32 GiB
RAM per cpu:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks=2 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G


(job_scripts_saga_accel)=

## Accel

*Accel* jobs are specified just like *normal* jobs except that they
also have to specify `--partition=accel`.  Also, they must also
specify how many GPUs to use, with `--gres=gpu:N`, where `N` is 1, 2,
3 or 4.

Here is an example that asks for 2 tasks and 2 gpus on one gpu node:

    #SBATCH --account=MyProject --job-name=MyJob
    #SBATCH --partition=accel --gres=gpu:2
    #SBATCH --time=1-0:0:0
    #SBATCH --ntasks-per-node=2 --nodes=1
    #SBATCH --mem-per-cpu=8G


(job_scripts_saga_devel)=

## Devel

_Devel_ jobs must specify `--qos=devel`.  A _devel_ job is like a _normal_
job, except that it has restrictions on job length and size.

For instance:

	#SBATCH --account=MyProject
	#SBATCH --job-name=MyJob
	#SBATCH --qos=devel
	#SBATCH --time=00:30:00
	#SBATCH --ntasks=16


(job_scripts_saga_optimist)=

## Optimist

_Optimist_ jobs are specified just like _normal_ jobs, except that
they also must must specify `--qos=optimist`.  They can run on any
node on Saga.

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
	#SBATCH --mem-per-cpu=3G
	#SBATCH --ntasks=16
	#SBATCH --time=2:00:00
