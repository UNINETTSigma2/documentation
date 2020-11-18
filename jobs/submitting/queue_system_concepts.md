# Queue System Concepts

This page explains some of the core concepts of the Slurm queue system.

For an overview of the Slurm concepts, Slurm has a beginners guide:
[Quick Start](https://slurm.schedmd.com/quickstart.html).

## Partition
The nodes on a cluster is divided into sets, called _partitions_.  The
partitions can be overlapping.

Several [job types](choosing_job_types.md) on our clusters are
implemented as partitions, meaning that one specifies `--partition` to
select job type -- for instance _bigmem_, _accel_ and _optimist_.

## QoS - Quality of Service
A _QoS_ is a way to assign properties and limitations to jobs.  It can
be used to give jobs different priority, and add or change the
limitations on the jobs, for instance the size or lenght of jobs, or
the number of jobs running at one time.

Several [job types](choosing_job_types.md) on our clusters are
implemented as a QoS, meaning that one specifies `--qos` to select
job type -- for instance _preproc_, _devel_ and _short_.
The jobs will then (by default) run in the standard (_normal_) partition,
but have different properties.

## Account
An _account_ is an entity that can be assigned a quota for resource
usage.  All jobs run in an account, and the job's usage is subtracted
from the account's quota.

Accounts can also have restrictions, like how man jobs can run in it
at the same time, or which reservations its jobs can use.

On our cluster, each project has its own account, with the same name
"nnXXXXk".  Some projects also have an account "nnXXXXo" for running
_optimist_ jobs.  We use accounts mainly for accounting resource
usage.

Read more about [projects and usage accounting](projects_accounting.md).

## Jobs
Jobs are submitted to the job queue, and starts running on assigned
compute nodes when there are enough resources available.

### Job step
A job is divided into one or more _job steps_.  Each time a job runs
`srun` or `mpirun`, a new job step is created.  Job steps are
[normally](guides/running_job_steps_parallel.md) executed
sequentially, one after each other.  In addition to these, the
batch job script itself, which runs on the first of the allocated
nodes, is considered a job step (`batch`).

`sacct` will show the job steps.  For instance:

	$ sacct -j 357055
		   JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
	------------ ---------- ---------- ---------- ---------- ---------- --------
	357055              DFT     normal    nn9180k        256  COMPLETED      0:0
	357055.batch      batch               nn9180k         32  COMPLETED      0:0
	357055.exte+     extern               nn9180k        256  COMPLETED      0:0
	357055.0      pmi_proxy               nn9180k          8  COMPLETED      0:0

The first line here is the job allocation.  Then comes the job script
step (`batch`), and an artificial step that we can ignore here
(`extern`), and finally a job step corresponding to an `mpirun` or
`srun` (step 0).  Further steps would be numbered 1, 2, etc.

### Tasks
Each job step starts one or more _tasks_, which corresponds to
processes.  So for instance the processes (mpi ranks) in an mpi job
step are tasks.  This is why one specifies `--ntasks` etc in job
scripts to select the number of processes to run in an mpi job.

Each task in a job step is started at the same time, and they run in
parallel on the nodes of the job.  `srun` and `mpirun` will take care
of starting the right number of processes on the right nodes.

(Unfortunately, Slurm also calls the individual instances of an [array
job](job_scripts/array_jobs.md) for _array tasks_.)
