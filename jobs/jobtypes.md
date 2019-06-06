# Job Types

To ensure efficient use of the compute nodes and give users fair access to the
resources, jobs are divided into different *types*, which determine where
and how they are run.  The different types are

- **normal**: The default type.  Most jobs are `normal` jobs.  They run
  on between 4 and 32 standard nodes (but see below).
- **bigmem**: Jobs that require more memory (RAM).  `bigmem` jobs run on
  special nodes with more RAM.
- **preproc**: Short, 1-node jobs for things like preprocessing and
  postprocessing, that don't need many cpus.
- **devel**: Short jobs, meant for quick development and testing.  They get
  higher priority to start sooner, but can only use upto 10 nodes each, and 16
  in total.
- **optimist** : Low priority jobs that can start whenever there are
  free resources, but will be requeued when other jobs need the
  resources.  Meant for jobs that use checkpointing.  Projects need to
  make a request for using this job type, and will get a separate
  quota for it.

## `normal` Jobs

`normal` jobs must specify account (`--account`), walltime limit (`--time`)
and number of nodes (`--nodes`). By default, they must use between 4 and 32
nodes, but a project can be allowed to use more nodes if needed.  The jobs can
specify how many tasks should run per node and how many CPUs should be used by
each task.

If more than 32 nodes is needed, and the application in question can actually 
scale more than 32 nodes, please send a request to <support@metacenter.no>.

All normal jobs gets exclusive access to whole nodes (all CPUs and memory).
If a job tries to use more (resident) memory than is configured on the nodes,
it will be killed.  Currently, this limit is 60 GiB, *but it can change*. 
If a job would require more memory per core than the given 60GiB split by 32 cores, the trick is to limit the number of cores/node the following way:


	#SBATCH --account=nn9999k 
	#SBATCH --time=1-0:0:0
	#SBATCH --nodes=10 --ntasks-per-node=4

The example above will use only 4 cores/node, giving each core 15GiB/core to the relevant job. If your job needs more than 60GiB/core, the only option on Fram is `--partition=bigmem`(see below). 

Maximal wall time limit for normal jobs is 7 days (168 hours). **With respect to walltime settings we recommend you to be as precise as you can when specifying the parameters as they will inflict on how fast your jobs will start to run. But: too long is better than too short due to lost work!**

Note that setting `--cpus-per-task` does *not* bind the tasks to the given number of
CPUs for normal jobs; it merely sets `$OMP_NUM_THREADS` so that OpenMP jobs by
default will use the right number of threads. (It is possible to override this
number by setting `$OMP_NUM_THREADS` in the job script.)

The [Sample MPI Batch Script](sample_mpi_script.md) page has an example of a normal MPI job.

## `bigmem` Jobs

There are eight nodes with 512 GiB memory and two nodes with 6 TiB. 
To use the large memory nodes, jobs must specify `--partition=bigmem`.  In
addition, they must specify wall time limit, number of CPUs/nodes and
memory memory per CPU. A `bigmem` job is assigned the requested CPUs and
memory exclusively, but shares nodes with other jobs. If a `bigmem` job tries
to use more resident memory than requested, it gets killed. The maximal wall
time limit for bigmem jobs is 14 days.

Here is an example that asks for 2 nodes, 3 tasks per node, and 4 CPUs per
task, and 32 GiB RAM per CPU:

    #SBATCH --account=nn9999k --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --nodes=2 --ntasks-per-node=3 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

Note that even though the memory specification is called `--mem-per-cpu`, the
memory limit the job gets on the node is for the total usage by all processes
on the node, so in the above example, it would get a limit of 3 * 4 * 32 GiB =
384 GiB. The queue system doesn't care how the memory usage is divided between
the processes or threads, as long as the total usage on the node is below the
limit.

Also note that contrary to `normal` jobs, `bigmem` jobs will be bound to the
CPU cores they are allocated, so the above sample job will have access to 12
cores on each node. However, the three tasks are free to use all cores the job
has access to on the node (12 in this example).

## `preproc` Jobs

`preproc` jobs must specify `--qos=preproc`.  A `preproc` job is allocated 1 node,
exclusively, and has a maximum walltime of 1 day.  Otherwise, it is the same
as a normal job.  The idea is that preprocessing, postprocessing or similar
doesn't need to use many nodes.  Note that there is a limit of how many
`preproc` jobs a project can run at the same time.

Example of a general `preproc` specification (1 node, 4 tasks with 8 CPUs/task):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0
    #SBATCH --ntasks-per-node=4 --cpus-per-task=8

Here is a simpler `preproc` job (one task on one node):

    #SBATCH --account=nn9999k --qos=preproc
    #SBATCH --time=1:0:0

## `devel` Jobs

`devel` jobs must specify `--qos=devel`.  A `devel` job is like a `normal`
job, except that it can use between 1 and 4 nodes, and there is a limit of
how many nodes `devel` jobs can use at the same time (currently 4),each user is
allowed to run only 1 `devel` job simultaneously.  `devel` jobs also have a 
maximum walltime of 30 minutes. On the other hand, they get higher priority than
other jobs, in order to start as soon as possible.

If you have temporary development needs that cannot be fulfilled by the devel 
QoS, please contact us at <support@metacenter.no>.


## `optimist` Jobs

`optimist` jobs must specify `--qos=optimist`, and should specify the
number of nodes needed (between 4 and 32), but they should *not*
specify wall time limit.  They run on the same nodes as `normal` jobs.
They get lower priority than other jobs, but can start as soon as
there are free resources.  However, when any other job needs its
resources, the `optimist` job is stopped and put back on the job
queue.  Therefore, all `optimist` jobs must use checkpointing, and
access to run `optimist` jobs will only be given to projects that
demonstrate that they can use checkpointing.

An `optimist` job will only be scheduled if there are free resources
at least 30 minutes when the job is considered for scheduling.
However, it can be requeued before 30 minutes have passed, so there is
no gurarantee of a minimum run time.  When an `optimist` job is
requeued, it is first sent a `SIGTERM` signal.  This can be trapped in
order to trigger a checkpoint.  After 30 seconds, the job receives a
`SIGKILL` signal, which cannot be trapped.

To be able submit `optimist` jobs the project has to send a request to
Sigma2 to get an optimist quota.

# Job Priorities

The priority setup has been designed to be as predictable and easy to
understand as possible, while trying to maximize the utilisation of the
cluster.

The principle is that a job's priority increases 1 point per minute the job is
waiting in the queue, and once the priority reaches 20,000, the job gets a
reservation in the future.  It can start before that, and before it gets a
reservation, but only if it does not delay any jobs with reservations.

The different job types start with different priorities:

- `devel` jobs start with 20,000, so get a reservation directly
- `normal`, `bigmem` and `preproc` jobs start with 19,760, so get a
  reservation in 4 hours
- "Unpri" `normal`, `bigmem` and `preproc` jobs(*) start with 19,040, so get a reservation in 16 hrs
- `optimist` jobs start with 1 and end at 10,080, so they never get a reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other lower priority job should be able delay it. But before
it gets a reservation, the queue system backfiller is free to start any other
job that can start now, even if that will delay the job.  Note that there are
still factors that can change the estimated start time, for instance running
jobs that exit sooner than expected, or nodes failing.

(*) In some cases when a project applies for extra CPU hours because it has
spent its allocation, it will get "unprioritized" hours, meaning their jobs
will have lower priority than standard jobs (but still higher than `optimist`
jobs).

