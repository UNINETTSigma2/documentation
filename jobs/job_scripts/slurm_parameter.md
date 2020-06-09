# SLURM Parameter and Settings

SLURM supports a multitude of different parameters. This enables you to
effectively tailor your script to your need but also
means that is easy to get lost and waste your time and quota.

The following parameters can be used as command line parameters with
`sbatch` and `srun` or in jobscripts. To use it in a jobscript, start
a newline with `#SBATCH` followed by the parameter. Replace <....>
with the value you want, e.g. `--job-name=test-job`.

## SLURM Parameter

### Basic settings:

| Parameter                            | Function
| ------------------------------------ | ------------------------
| `--job-name=<name>`              | Job name to be displayed by for example `squeue`
| `--output=<path>`                | Path to the file where the job (error) output is written to
| `--mail-type=<type>`             | Turn on mail notification; type can be one of BEGIN, END, FAIL, REQUEUE or ALL
| `--mail-user=<email_address>`   | Email address to send notifications to

### Requesting Resources

| Parameter                              | Function
| -------------------------------------- | -------------------------------
| `--time=<d-hh:mm:ss>`              | Time limit for job. Job will be killed by SLURM after time has run out. Format days-hours:minutes:seconds
| `--nodes=<num_nodes>`             | Number of nodes. Multiple nodes are only useful for jobs with distributed-memory (e.g. MPI).
| `--mem=<MB>`                       | Memory (RAM) per node. Number followed by unit prefix, e.g. 16G
| `--mem-per-cpu=<MB>`               | Memory (RAM) per requested CPU core
| `--ntasks-per-node=<num_procs>`   | Number of (MPI) processes per node. More than one useful only for MPI jobs. Maximum number depends nodes (number of cores)
| `--cpus-per-task=<num_threads>`   | CPU cores per task. For MPI use one. For parallelized applications benchmark this is the number of threads.
| `--exclusive`                            | Job will not share nodes with other running jobs. You will be charged for the complete nodes even if you asked for less.

### Accounting

| Parameter                  | Function
| -------------------------- | -----------------------------
| `--account=<name>`     | Project (not user) account the job should be charged to.
| `--partition=<name>`   | Partition/queue in which to run the job.
| `--qos=<devel/short>`                | The *devel* or *short* QOS (quality of servive) can be used to submit short jobs for testing and debugging.

See also [projects and accounting](projects.md) for more information.

SLURM differs slightly from the previous Torque system with respect to
definitions of various parameters, and what was known as queues in
Torque may be covered by either `--partition=...` or `--qos=...`.

Check our cluster specific sites for an overview of the partitions and
QOS of that system:

- [Fram](fram_job_types.md)
- [Saga](saga_job_types.md)

### Advanced Job Control

| Parameter                          | Function
| ---------------------------------- | ---------------------------
| `--array=<indexes>`            | Submit a collection of similar jobs, e.g. `--array=1-10`. (sbatch command only). [More info](array_jobs.md)
| `--dependency=<state:jobid>`   | Wait with the start of the job until specified dependencies have been satified. E.g. --dependency=afterok:123456
| `--ntasks-per-core=2`                | Enables hyperthreading. Only useful in special circumstances.


## Differences between CPUs and tasks

As a new users writing your first SLURM job script the difference
between `--ntasks` and `--cpus-per-taks` is typically quite confusing.
Assuming you want to run your program on a single node with 16 cores
which SLURM parameters should you specify?

The answer is it depends whether the your application supports MPI. MPI
(message passing protocol) is a communication interface used for
developing parallel computing programs on distributed memory systems.
This is necessary for applications running on multiple computers (nodes)
to be able to share (intermediate) results.

To decide which set of parameters you should use, check if your
application utilizes MPI and therefore would benefit from running on
multiple nodes simultaneously. On the other hand you have an non-MPI
enables application or made a mistake in your setup, it doesn't make
sense to request more than one node.

## Settings for OpenMP and MPI jobs

### Single node jobs

For applications that are not optimized for HPC (high performance
computing) systems like simple python or R scripts and a lot of software
which is optimized for desktop PCs.

#### Simple applications and scripts

Many simple tools and scripts are not parallized at all and therefore
won't profit from more than one CPU core.

| Parameter             | Function
| --------------------- | ----------------------------------------
| `--nodes=1`             | Start a unparallized job on only one node
| `--ntasks-per-node=1`   | For OpenMP, only one task is necessary
| `--cpus-per-task=1`     | Just one CPU core will be used.
| `--mem=<MB>`      | Memory (RAM) for the job. Number followed by unit prefix, e.g. 16G

If you are unsure if your application can benefit from more cores try a
higher number and observe the load of your job. If it stays at
approximately one there is no need to ask for more than one.

#### OpenMP applications

OpenMP (Open Multi-Processing) is a multiprocessing library is often
used for programs on shared memory systems. Shared memory describes
systems which share the memory between all processing units (CPU cores),
so that each process can access all data on that system.

| Parameter                              | Function
| -------------------------------------- | ---------------------------
| `--nodes=1`                              | Start a parallel job for a shared memory system on only one node
| `--ntasks-per-node=1`                    | For OpenMP, only one task is necessary
| `--cpus-per-task=<num_threads>`   | Number of threads (CPU cores) to use
| `--mem=<MB>`                       | Memory (RAM) for the job. Number followed by unit prefix, e.g. 16G

### Multiple node jobs (MPI)

For MPI applications.

Depending on the frequency and bandwidth demand of your setup, you can
either just start a number of MPI tasks or request whole nodes. While
using whole nodes guarantees that a low latency and high bandwidth it
usually results in a longer queuing time compared to cluster wide job.
With the latter the SLURM manager can distribute your task across all
nodes of stallo and utilize otherwise unused cores on nodes which for
example run a 16 core job on a 20 core node. This usually results in
shorter queuing times but slower inter-process connection speeds.

This will make the best use of the resources and give the most
predictable execution times. If your job requires more than the default
available memory per core (32 GB/node gives 2 GB/core for 16 core nodes
and 1.6GB/core for 20 core nodes) you should adjust this need with the
following command: `#SBATCH --mem-per-cpu=4GB` When doing this, the
batch system will automatically allocate 8 cores or less per node.

#### To use whole nodes

| Parameter                              | Function
| -------------------------------------- | ---------------------------
| `--nodes=<num_nodes>`             | Start a parallel job for a distributed memory system on several nodes
| `--ntasks-per-node=<num_procs>`   | Number of (MPI) processes per node. Maximum number depends on nodes
| `--cpus-per-task=1`                      | Use one CPU core per task.
| `--exclusive`                            | Job will not share nodes with other running jobs. You don't need to specify memory as you will get all available on the node.

#### To distribute your job

| Parameter                     | Function
| ----------------------------- | ----------------------------------
| `--ntasks=<num_procs>`   | Number of (MPI) processes in total. Equals to the number of cores
| `--mem-per-cpu=<MB>`      | Memory (RAM) per requested CPU core. Number followed by unit prefix, e.g. 2G

### Scalability

You should run a few tests to see what is the best fit between
minimizing runtime and maximizing your allocated cpu-quota. That is you
should not ask for more cpus for a job than you really can utilize
efficiently. Try to run your job on 1, 2, 4, 8, 16, etc., cores to see
when the runtime for your job starts tailing off. When you start to see
less than 30% improvement in runtime when doubling the cpu-counts you
should probably not go any further. Recommendations to a few of the most
used applications can be found in sw\_guides.


### Troubleshooting

#### "srun: Warning: can't honor --ntasks-per-node set to _X_ which doesn't match the requested tasks _Y_ with the number of requested nodes _Y_. Ignoring --ntasks-per-node."

This warning appears when using the `mpirun` command with Intel MPI and
specifying `--ntasks-per-node` for jobs in the `normal` partition on Fram.  As
far as we have seen, the job does *not* ignore the `--ntasks-per-node`, and
will run the specified number of processes per node.  You can test it with,
e.g., `mpirun hostname`.  Please let us know if you have an example where
`--ntasks-per-node` is *not* honored!

So, if you get this when using `mpirun` with Intel MPI, our recommendation is
currently that the warning can be ignored.
