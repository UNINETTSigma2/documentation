# Job Types

To ensure the efficient use of the compute nodes, jobs can ask which partition it should be run on, the memory size, and set a running time expectation.
Slurm then uses these information to assign and allocate resources.

## Partitions

Partitions are logical groupings of resources so that each partition can be set up specific to a job type. The partition can be thought of
as queues and different queues are meant to run different job types. A job that requires more memory can be in a different partition than a job that does not require a large amount of memory.

These are the main job partitions:

1. **normal**: Most production jobs, runs on all standard compute nodes.
2. **bigmem**: Jobs on the _bigmem_ nodes.
3. **optimist**: Low priority jobs that can be requeued when higher priority jobs need their nodes. They must implement checkpointing.
4. **accel**: Jobs on compute nodes with GPU cards. **Not implemented yet**

To set the partition, use `--partition=` in the job script file. For example:

    #SBATCH --partition=normal

For more information, visit Slurm's [Quality of Service](https://slurm.schedmd.com/qos.html) documentation.

### **normal** Jobs

Normal jobs must specify account, walltime limit and number of nodes. By
default, jobs must use between 4 and 30 nodes, but a project can be allowed to
use more nodes if needed. It can also specify how many tasks should run per
node and how many CPUs should be used by each task.

All normal jobs gets exclusive access to whole nodes (all CPUs and memory).
If a job tries to use more (resident) memory than is configured on the nodes,
it will be killed. Currently, this limit is 60 GiB, *but could change*. The
maximal wall time limit for normal jobs is two days (48 hours).

Note that setting `--cpus-per-task` does *not* bind the tasks to the given number of
CPUs for normal jobs; it merely sets `$OMP_NUM_THREADS` so that OpenMP jobs by
default will use the right number of threads. It is possible to override this
number by setting `$OMP_NUM_THREADS` in the job script.

The [Sample Batch Script](samplescript.md) page has an example of a normal job.

### **bigmem** Jobs

To use the **bigmem** nodes, `--partition=bigmem`, wall time limit, CPUs and
memory must be specified. There are eight nodes with 512 GiB memory and two nodes
with 6 TiB. A bigmem job is assigned the requested CPUs and
memory exclusively, but shares nodes with other jobs. If a bigmem job tries
to use more resident memory than requested, it gets killed. The maximal wall
time limit for bigmem jobs is 14 days.

Here is an example that asks for 2 nodes, 2 tasks per node, and 4 cpus per
task:

    #SBATCH --account=nn9999k --partition=bigmem
    #SBATCH --time=1-0:0:0
    #SBATCH --nodes=2 --ntasks-per-node=2 --cpus-per-task=4
    #SBATCH --mem-per-cpu=32G

Note that even though the memory specification is called `--mem-per-cpu`, the
memory limit the job gets on the node is for the total usage by all processes
on the node, so in the above example, it would get a limit of 2 * 4 * 32 GiB =
256 GiB. The queue system doesn't care how the memory usage is divided
between the processes or threads, as long as the total usage is below the
limit.

Also note that contrary to *normal* jobs, *bigmem* jobs will be bound to the
CPU cores they are allocated, so the above sample job will have access to 8 cores on
each node. However, the two tasks are free to use all cores the job has
access to on the node (8 in this example).

### **preproc** Jobs

**preproc** is a special partition set up for preprocessing jobs (set `--partition=preproc` in job script file). A
preproc job is allocated 1 node, exclusively. Otherwise, it is the same as a normal
job. The idea is that preprocessing or similar doesn't need to use many
nodes. Note that there is a limit of how many preproc jobs a project can
run at the same time.

To run a preproc job, set `--partition=preproc` in the script file. It is not necessary
to specify the number of nodes but if specified, it must be 1.)

Example of a general preproc specification (1 node, 4 tasks with 8 CPUs/task):

    #SBATCH --account=nn9999k --partition=preproc
    #SBATCH --time=1:0:0
    #SBATCH --ntasks-per-node=4 --cpus-per-task=8

Here is a simpler preproc job (one task on one node):

    #SBATCH --account=nn9999k --partition=preproc
    #SBATCH --time=1:0:0

## Node Information  {#nodeinfo}

There are a few commands for getting information about nodes:

- List all nodes in all partitions: `sinfo`
- List nodes in `normal` partition: `sinfo -p normal`
- List a specific node: `sinfo -n` *nodename*
- See all info about a node: `scontrol show node` *nodename*

`sinfo` has a lot of options for selecting output fields; see `man sinfo` for
details.

## Job priority

The priority setup has been designed to be predictable and easy to
understand, and to try and maximize the utilisation of the cluster.

-   Principle: *Time-to-reservation*
-   Priority increases 1 point/minute
-   When priority is >= 20,000, job gets a reservation
-   **Devel** jobs start with 19,990, so reservation in 10 min
-   **Normal** jobs start with 19,940, so reservation in 60 min
-   **Normal unpri** jobs start with 19,400, so reservation in 10 hrs
-   **Optimist** jobs start with 1, ends at 10,080, so never reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other job should delay it. But before it gets a reservation,
the queue system backfiller is free to start any other job that can start now,
even if that will delay the job.
