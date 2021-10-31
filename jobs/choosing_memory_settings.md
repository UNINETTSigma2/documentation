(choosing-memory-settings)=

# How to choose the right amount of memory

```{warning}
**Do not ask for a lot more memory than you need**

A job that asks for many CPUs but little memory will be billed
for its number of CPUs, while a job that asks for a lot of memory but
few CPUs, may be billed for its memory requirement.

If you ask for a lot more memory than you need, you might be surprised
that your job will bill your project a lot more than you expected.

If you ask for a lot more memory than you need, your job may queue much
longer than it would asking for less memory.
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
- Your compute account gets charged too much (problem for you and your
  allocation group, as well as for the tax payer financing these compute
  resources)
- Other users have to wait longer with their jobs (problem for us and for other users)
- Your jobs may queue much longer (problem for you) because the scheduler does not know that you
  might use a lot less memory than you ask it to reserve for you

It is important to make sure that your jobs use the right amount of memory and
the right number of CPUs in order to help you and others using the HPC machines
utilize these resources more efficiently, and in turn get work done more
speedily.

If you ask for too little memory, your job will be stopped and it might be
stopped late in the run.

We recommend users to run a test job before submitting many similar runs to the queue system
and find out how much memory is used (see below for examples on how to do that).
Once you know, add perhaps 15-20% extra memory (and runtime) for the job compared to what
your representative test case needed.

Remember to check the [Slurm documentation](https://slurm.schedmd.com/squeue.html#lbAG),
[job types](choosing_job_types.md),
[queue system concepts](submitting/queue_system_concepts.md),
and [HPC machines](/hpc_machines/hardware_overview.md)
to verify that you are submitting the right job to the right partition and
right hardware.

Speaking of right partition, one way to get more memory if a node is not enough
is to spread the job over several nodes by asking for more cores than needed. But this comes
at the price of paying for more resources, queuing longer, and again blocking others.
A good alternative is often to get access to "highmem" nodes which are designed for jobs
with high memory demand.


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

    print *, 'will try to allocate', size_mb, 'MB'

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
    print *, 'ok all went fine'

end program
```

Compile it like this and later we can examine `mybinary` and check whether we can find out that
it really allocated 3210 MB:
```console
$ gfortran example.f90 -o mybinary
```


### By using the Slurm browser

This approach only works for a currently running job.

Log into one of the available Slurm browsers [here](monitoring.md).
You need to be on the university network to reach these. Use a VPN solution
(check with your university for details on how to connect using VPN) if you
would like to access these from home or outside.

Once connected, you can have a look at all currently running jobs. There is a
lot of interesting data but here we focus on the memory part.

In this example you can see that the node has a bit above 60 GB memory and the
job consumes about half at the beginning of the run and later oscillates
consuming around a third of the available memory. But you can also see that the
memory consumption varies over time.

![Memory high water mark in the Slurm browser](img/slurmbrowser-memory.jpg "Memory high water mark in the Slurm browser")


### Using top

While the job is running, find out on which node(s) it runs using `squeue -u $USER`,
then `ssh` into one of the listed compute nodes and run `top -u $USER`.


### By checking the Slurm output generated with your job

We can use the following example script (adapt `--account=nn____k`; this is tested on Saga):
```{code-block}
---
emphasize-lines: 3-3
---
#!/bin/bash

#SBATCH --account=nn____k
#SBATCH --qos=devel

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --ntasks=1

# we could also compile it outside of the job script
gfortran example.f90 -o mybinary

./mybinary
```

Slurm generates an output for each job you run. For instance job number `4200691`
generated output `slurm-4200691.out`.

This output contains the following:
```{code-block}
---
emphasize-lines: 12-12
---
Task and CPU usage stats:
       JobID    JobName  AllocCPUS   NTasks     MinCPU MinCPUTask     AveCPU    Elapsed ExitCode
------------ ---------- ---------- -------- ---------- ---------- ---------- ---------- --------
4200691      mem-profi+          1                                             00:00:37      0:0
4200691.bat+      batch          1        1   00:00:01          0   00:00:01   00:00:37      0:0
4200691.ext+     extern          1        1   00:00:00          0   00:00:00   00:00:37      0:0

Memory usage stats:
       JobID     MaxRSS MaxRSSTask     AveRSS MaxPages   MaxPagesTask   AvePages
------------ ---------- ---------- ---------- -------- -------------- ----------
4200691
4200691.bat+   3210476K          0   3210476K        0              0          0
4200691.ext+          0          0          0        0              0          0

Disk usage stats:
       JobID  MaxDiskRead MaxDiskReadTask    AveDiskRead MaxDiskWrite MaxDiskWriteTask   AveDiskWrite
------------ ------------ --------------- -------------- ------------ ---------------- --------------
4200691
4200691.bat+        1.11M               0          1.11M        0.02M                0          0.02M
4200691.ext+        0.00M               0          0.00M            0                0              0
```

From this (see below `Memory usage stats`) we can find out that the job needed
`3210476K` memory (3210476 KB).

Note that Slurm samples the memory every 30 seconds. This means that if your
job is shorter than 30 seconds, it will show that your calculation consumed
zero memory which is probably wrong.  The sampling rate also means that if your
job contains short peaks of high memory consumption, the sampling may
completely miss these.


#### Slurm reports values for each job step

In the [Slurm manual for sacct](https://slurm.schedmd.com/sacct.html) the
following is stated for `AvgRSS`:  *Average resident set size of all tasks in
job*. And similarly for `MaxRSS` and all other `Max<something>` or
`Ave<something>` values: These values give information for each *step* of a
job, each being a call to `srun` or `mpirun` in your job script.  Also, the job
script itself (commands run in the jobscript without using `srun` or `mpirun`)
is counted as a *step*, called the `batch` step.  The *average* or *maximum* is
calculated over the tasks (processes) in each step.  Meaning, if you call
`srun` or `mpirun` multiple times in your job script they will get one line
each in the `sacct` output with separate entries for both `AvgRSS` and
`MaxRSS`.


### By using sacct

This creates a short version of the above.

As an example, I want to know this for my job which had the number `4200691`:
```console
$ sacct -j 4200691 --format=MaxRSS

    MaxRSS
----------

  3210476K
         0
```

From this we see that the job needed `3210476K` memory, same as above. The
comment above about possibly multiple steps applies also here.


### By prepending your binary with /usr/bin/time -v

In your job script instead of running `./mybinary` directly, prepend it with `/usr/bin/time -v`:
```{code-block}
---
emphasize-lines: 15-15
---
#!/bin/bash

#SBATCH --account=nn____k
#SBATCH --qos=devel

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:00
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
User time (seconds): 0.32
System time (seconds): 0.67
Percent of CPU this job got: 2%
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:36.00
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 3210660
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 2
Minor (reclaiming a frame) page faults: 2559
Voluntary context switches: 19
Involuntary context switches: 3
Swaps: 0
File system inputs: 1073
File system outputs: 0
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0
```

The relevant information in this context is `Maximum resident set size
(kbytes)`, in this case 3210660 kB which is what we expected to find.  Note
that it has to be `/usr/bin/time -v` and `time -v` alone will not do it.


### By using Arm Performance Reports

You can profile your job using {ref}`Arm Performance Reports <arm-performance-reports>`.

Here is an example script (adapt `--account=nn____k`; this is tested on Saga):
```{code-block}
---
emphasize-lines: 11-13, 15
---
#!/bin/bash

#SBATCH --account=nn____k
#SBATCH --qos=devel

#SBATCH --job-name='mem-profiling'
#SBATCH --time=0-00:01:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --ntasks=1

module purge
module load Arm-Forge/21.1

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

And gradually reduce it until your job fails with `oom-kill` ("oom" is short for "out of memory"):
```
slurm_script: line 11: 33333 Killed ./mybinary
slurmstepd: error: Detected 1 oom-kill event(s) in step 997857.batch cgroup.
Some of your processes may have been killed by the cgroup out-of-memory
handler.
```

Or you start with a very conservative estimate and you gradually increase until
the job is not stopped.

Then you also know. But there are more elegant ways to find this out (see
options above).
