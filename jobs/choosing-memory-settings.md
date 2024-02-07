(choosing-memory-settings)=

# How to choose the right amount of memory

```{admonition} Do not ask for a lot more memory than you need

A job that asks for many CPUs but little memory will be billed
for its number of CPUs, while a job that asks for a lot of memory but
few CPUs, may be billed for its memory requirement.

If you ask for a lot more memory than you need:
- You might be surprised that your job will bill your project a lot more than
  you expected.
- Your job may queue much longer than it would asking for less memory.
- The possibility to parallelize your job/workflow may be more limited.
```

```{contents} Table of Contents
    :depth: 3
```


## Why it matters

Please read the above box, please share this with your colleagues and students.
A significant portion of resources risks to be wasted if you ask for too much memory.

This is because memory and CPU are not completely independent. If I ask for too
much memory, I also block the compute resources which carry that memory
resource at the same time.

A too generous "better safe than sorry" approach to memory allocation leads to these problems:
- Your compute account **gets charged too much** (problem for you and your
  allocation group, as well as for the tax payer financing these compute
  resources)
- Other users have to **wait longer with their jobs** (problem for us and for other users)
- Your **jobs may queue much longer** (problem for you) because the scheduler does
  not know that you might use a lot less memory than you ask it to reserve for you
- The **possibility to parallelize your job/workflow may be severely limited** if
  you ask for excessive amount of unused memory since there may not be enough
  memory left for parallel jobs running at the same time

It is important to make sure that your jobs use the right amount of memory
(below we show how to find out) and {ref}`the right number of CPUs
<choosing-number-of-cores>` in order to help you and others using the HPC
machines utilize these resources more efficiently, and in turn get work done
more speedily.

If you ask for too little memory, your job will be stopped and it might be
stopped late in the run.


## Run a test job before running many similar jobs

We recommend users to run a test job before submitting many similar runs to the
queue system and find out how much memory is used (see below for examples on
how to do that).  **Once you know, add perhaps 15-20% extra memory** (and runtime)
for the job compared to what your representative test case needed.

Remember to check the [Slurm documentation](https://slurm.schedmd.com/squeue.html#lbAG),
[job types](choosing_job_types.md),
[queue system concepts](submitting/queue_system_concepts.md),
and [HPC machines](/hpc_machines/hardware_overview.md)
to verify that you are submitting the right job to the right partition and
right hardware.


## How to get more memory if you need it

Speaking of right partition, one way to get more memory if a node is not enough
is to spread the job over several nodes by asking for more cores than needed.
But this comes at the price of paying for more resources, queuing longer, and
possibly blocking others.  A good alternative for jobs that need a lot of memory
is often to get access to "highmem" nodes which are designed for jobs with high
memory demand.


## How to find out how much memory you need


### Example code which you can use for testing this


You can test some of the approaches with the following example code (`example.f90`)
which allocates 3210 MB (you can adapt that value if you want it to consume
more or less memory):

```{code-block} fortran
---
emphasize-lines: 5-5
---
program example

    implicit none

    integer(8), parameter :: size_mb = 3210
    integer(8) :: size_mw
    real(8), allocatable :: a(:)

!   print *, 'will try to allocate', size_mb, 'MB'

    size_mw = int(1000000*size_mb/7.8125d0)
    allocate(a(size_mw))

    ! this is here so that the allocated memory gets actually used and the
    ! compiler does not skip allocating the array
    a = 1.0d0
    print *, 'first element:', a(1)

    ! wait for 35 seconds
    ! because slurm only samples every 30 seconds
    call sleep(35)

    deallocate(a)
    print *, 'ok all went well'

end program
```

Compile it like this and later we can examine `mybinary` and check whether we can find out that
it really allocated 3210 MB:
```console
$ gfortran example.f90 -o mybinary
```


### Using top

While the job is running, find out on which node(s) it runs using `squeue --me`,
then `ssh` into one of the listed compute nodes and run `top -u $USER`.


### By checking the Slurm output generated with your job

We can use the following example script (adapt `--account=nn____k`; this is tested on Saga):
```{code-block}
---
emphasize-lines: 3-3
---
#!/usr/bin/env bash

#SBATCH --account=nn____k

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:30
#SBATCH --mem-per-cpu=3500M
#SBATCH --ntasks=1

# we could also compile it outside of the job script
gfortran example.f90 -o mybinary

./mybinary
```

Slurm generates an output for each job you run. For instance job number `10404698`
generated output `slurm-10404698.out`.

This output contains the following:
```{code-block}
---
emphasize-lines: 14,17
---
Submitted 2024-02-04T13:11:03; waited 27.0 seconds in the queue after becoming eligible to run.

Requested wallclock time: 2.0 minutes
Elapsed wallclock time:   44.0 seconds

Task and CPU statistics:
ID              CPUs  Tasks  CPU util                Start  Elapsed  Exit status
10404698           1            0.0 %  2024-02-04T13:11:30   44.0 s  0
10404698.batch     1      1     2.7 %  2024-02-04T13:11:30   44.0 s  0

Used CPU time:   1.2 CPU seconds
Unused CPU time: 42.8 CPU seconds

Memory statistics, in GiB:
ID               Alloc   Usage
10404698           3.4
10404698.batch     3.4     3.1
```

From this (see below `Memory statistics`) we can find out that the job used 3.1
GiB memory.

Note that **Slurm samples the memory every 30 seconds**. This means that if your
job is shorter than 30 seconds, it will show that your calculation consumed
zero memory which is probably wrong.  The sampling rate also means that if your
job contains short peaks of high memory consumption, the sampling may
miss these.


#### Slurm reports values for each job step

If you call `srun` or `mpirun` multiple times in your job script they will get
one line each in the `sacct` output with separate entries for both `AvgRSS` and
`MaxRSS`.

Also the job script itself (commands run in the jobscript without using `srun`
or `mpirun`) is counted as a *step*, called the `batch` step.


### By using sacct

This creates a short version of the above.

As an example, I want to know this for my job which had the number `10404698`:
```console
$ sacct -j 10404698 --format=MaxRSS

    MaxRSS
----------

  3210588K
         0
```

From this we see that the job needed `3210588K` memory, same as above. The
comment above about possibly multiple steps applies also here.


### Using jobstats

`jobstats` is always run as part of your job output. But knowing that the
command exists also on its own can be useful if you still know/remember the job
number (here: 10404698) but you have lost the Slurm output.

```console
$ jobstats -j 10404698
Job 10404698 consumed 0.0 billing hours from project nn****k.

Submitted 2024-02-04T13:11:03; waited 27.0 seconds in the queue after becoming eligible to run.

Requested wallclock time: 2.0 minutes
Elapsed wallclock time:   44.0 seconds

Task and CPU statistics:
ID              CPUs  Tasks  CPU util                Start  Elapsed  Exit status
10404698           1            0.0 %  2024-02-04T13:11:30   44.0 s  0
10404698.batch     1      1     2.7 %  2024-02-04T13:11:30   44.0 s  0

Used CPU time:   1.2 CPU seconds
Unused CPU time: 42.8 CPU seconds

Memory statistics, in GiB:
ID               Alloc   Usage
10404698           3.4
10404698.batch     3.4     3.1
```


### Using seff

`seff` is a nice tool which we can use on **completed jobs**. For example here we ask
for a summary for the job number 10404698:

```{code-block} console
---
emphasize-lines: 11-12
---
$ seff 10404698

Job ID: 10404698
Cluster: saga
User/Group: someuser/someuser
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:01
CPU Efficiency: 2.27% of 00:00:44 core-walltime
Job Wall-clock time: 00:00:44
Memory Utilized: 3.06 GB
Memory Efficiency: 89.58% of 3.42 GB
```


### By prepending your binary with /usr/bin/time -v

In your job script instead of running `./mybinary` directly, prepend it with `/usr/bin/time -v`:
```{code-block}
---
emphasize-lines: 14
---
#!/usr/bin/env bash

#SBATCH --account=nn____k

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:30
#SBATCH --mem-per-cpu=3500M
#SBATCH --ntasks=1

# instead of this:
# ./mybinary

# we do this:
/usr/bin/time -v ./mybinary
```

Then in the Slurm output we find:
```{code-block}
---
emphasize-lines: 10-10
---
Command being timed: "./mybinary"
User time (seconds): 0.51
System time (seconds): 0.64
Percent of CPU this job got: 3%
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:36.16
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 3212160
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 1
Minor (reclaiming a frame) page faults: 2394
Voluntary context switches: 20
Involuntary context switches: 41
Swaps: 0
File system inputs: 57
File system outputs: 0
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0
```

The relevant information in this context is `Maximum resident set size
(kbytes)`, in this case 3210916 kB which is what we expected to find.  Note
that it has to be `/usr/bin/time -v` and `time -v` alone will not do it.


### By using Arm Performance Reports

You can profile your job using {ref}`Arm Performance Reports <arm-performance-reports>`.

Here is an example script (adapt `--account=nn____k`; this is tested on Saga):
```{code-block}
---
emphasize-lines: 11-12, 14
---
#!/usr/env/bin bash

#SBATCH --account=nn____k
#SBATCH --qos=devel

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --ntasks=1

module purge
module load Arm-Forge/22.1.3

perf-report ./mybinary
```

This generates a HTML and text summary. These reports contain lots of
interesting information. Here showing the relevant part of the text report for
the memory:
```{code-block}
---
emphasize-lines: 4-4
---
Memory:
Per-process memory usage may also affect scaling:
Mean process memory usage:                    1.39 GiB
Peak process memory usage:                    3.07 GiB
Peak node memory usage:                      27.0% |==|
```


### By reducing the memory parameter until a job fails

This is not an elegant approach but can be an OK approach to calibrate one
script before submitting 300 similar jobs.

What you can do is to start with a generous memory setting:
```
#SBATCH --mem-per-cpu=3500M
```

And gradually reduce it until your job fails with `oom-kill` ("**oom**" or "**OOM**" is short for "out of memory"):
```
slurmstepd: error: Detected 1 oom_kill event in StepId=10404708.batch.
Some of the step tasks have been OOM Killed.
```

Or you start with a very conservative estimate and you gradually increase until
the job is not stopped.

Then you also know. But there are more elegant ways to figure this out (see
options above).
