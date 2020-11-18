(arm-performance-reports)=

# How to check the performance and scaling using Arm Performance Reports

[Arm Performance Reports](https://developer.arm.com/docs/101137/latest/contents)
is a performance evaluation tool which is simple to use, produces a
clear, single-file report, and it is used to obtain a
high-level overview of the performance characteristics.

It can report CPU time spent on various types of instructions (e.g.,
floating-point), communication time (MPI), multi-threading level and
thread synchronization overheads, memory bandwidth, and IO performance.
Such a report can help spotting certain bottlenecks in the
code and highlight potential optimization directions, but also suggest
simple changes in how the code should be executed to better utilize
the resources. Some typical examples of the suggestions are

> The CPU performance appears well-optimized for numerical
computation. The biggest gains may now come from running at larger
scales.

or

> Significant time is spent on memory accesses. Use a profiler
to identify time-consuming loops and check their cache performance.

A successful Arm Performance Reports run will produce two files, a HTML summary
and a text file summary, like in this example:

```
example_128p_4n_1t_2020-05-23_18-04.html
example_128p_4n_1t_2020-05-23_18-04.txt
```


## Do I need to recompile the code?

- You can use Arm Performance Reports on dynamically linked binaries without
  recompilation.  However, you may have to recompile statically linked binaries
  (for this please consult the
  [official documentation](https://developer.arm.com/docs/101137/2003)).
- Due to a bug in older versions of OpenMPI, on Fram Arm Performance
  Reports works only with OpenMPI version 3.1.3 and newer. If you have
  compiled your application with OpenMPI 3.1.1, you don't need to
  recompile it. Simply load the 3.1.3 module - those versions are
  compatible.


## Profiling a batch script

Let us consider the following example job script as your
usual computation which you wish to profile:

```bash
#!/bin/bash -l

# all your SBATCH directives
#SBATCH --account=myaccount
#SBATCH --job-name=without-apr
#SBATCH --time=0-00:05:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

# recommended bash safety settings
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

srun ./myexample.x  # <- we will need to modify this line
```

To profile the code you don't need to modify any of the `#SBATCH` part.
All we need to do is to load the `Arm-PerfReports/20.0.3` module
and to modify the `srun` command to instead use
[perf-report](https://developer.arm.com/docs/101137/latest/running-with-an-example-program):

```bash
#!/bin/bash -l

# ...
# we kept the top of the script unchanged

# we added this line:
module load Arm-PerfReports/20.0.3

# we added these two lines:
echo "set sysroot /" > gdbfile
export ALLINEA_DEBUGGER_USER_FILE=gdbfile

# we added perf-report in front of srun
perf-report srun ./myexample.x
```

This works the same way on Saga, Fram, and Betzy.
In other words, add 3 lines, replace `srun` or `mpirun -n ${SLURM_NTASKS}` by
`perf-report srun`.

That's it.

Why are these two lines needed?
```bash
echo "set sysroot /" > gdbfile
export ALLINEA_DEBUGGER_USER_FILE=gdbfile
```
We have a Slurm plug-in that (deliberately) detaches a job from the global mount
name space in order to create private versions of `/tmp` and `/var/tmp` (i.e.,
bind mounted) for each job. This is done both so jobs cannot see other jobs'
`/tmp` and `/var/tmp`, and also so that we avoid filling up (the global) `/tmp`
and `/var/tmp` (since we allow more than one job per compute node, we cannot
clean these directories after each job - we don't know which job created the
files). However, for `perf-report` to work with this setup we need to set GDB's
sysroot to `/`.


## Profiling on an interactive compute node

To run interactive tests one needs to submit
[an interactive job](interactive_jobs.md)
to Slurm using `srun` (**not** using `salloc`), e.g.:

First obtain an interactive compute node (adjust "myaccount"), on Saga:
```bash
$ srun --nodes=1 --ntasks-per-node=4 --mem-per-cpu=1G --time=00:30:00 --qos=devel --account=myaccount --pty bash -i
```
or Fram:
```bash
$ srun --nodes=1 --ntasks-per-node=32 --time=00:30:00 --qos=devel --account=myaccount --pty bash -i
```
or Betzy:
```bash
$ srun --nodes=1 --ntasks-per-node=128 --time=00:30:00 --qos=devel --account=myaccount --pty bash -i
```

Once you get the interactive node, you can run the profile:
```bash
$ module load Arm-PerfReports/20.0.3
$ echo "set sysroot /" > gdbfile
$ export ALLINEA_DEBUGGER_USER_FILE=gdbfile
$ perf-report srun ./myexample.x
```


## Use cases and pitfalls

We demonstrate some pitfalls of profiling, and show how
one can use profiling to reason about the performance of real-world
codes.

- [STREAM benchmark](arm-perf/stream.md) (measures the memory bandwidth)
- [LINPACK benchmark](arm-perf/linpack.md) (measures the floating-point capabilities)
- [OSU benchmark](arm-perf/osu.md) (measures the interconnect performance)
- [Quantifying the profiling overhead](arm-perf/overhead.md)


## What if the job timed out?

The job has to finish within the allocated time for the report to be generated.
So if the job times out, there is a risk that no report is generated.

If you run a job that always times out by design (in other words the job never
terminates itself but is terminated by Slurm), there is a workaround **if you
are running the profile on Fram on no more than 64 cores**:

As an example let us imagine we profile the following example:

```bash
# ...
#SBATCH --time=0-01:00:00
# ...

module load Arm-PerfReports/20.0.3

echo "set sysroot /" > gdbfile
export ALLINEA_DEBUGGER_USER_FILE=gdbfile

perf-report srun ./myexample.x  # <- we will need to change this
```

Let's imagine the above example code (`./myexample.x`) always times out,
and we expect it to time out after 1 hour (see `--time` above).
In this case we get no report.

To get a report on Fram, we can do this instead:

```bash
# ...
#SBATCH --time=0-01:00:00
# ...

module load Arm-MAP/20.0.3  # <- we changed this line

echo "set sysroot /" > gdbfile
export ALLINEA_DEBUGGER_USER_FILE=gdbfile

# we changed this line and tell map to stop the code after 3500 seconds
map --profile --stop-after=3500 srun ./myexample.x
```

We told map to stop the code after 3500 seconds but to still have some time to
generate a map file. This run will generate a file with a `.map` suffix. From
this file we can generate the profile reports on the command line (no need to
run this as a batch script):

```bash
$ module load Arm-PerfReports/20.0.3
$ perf-report prefix.map               # <- replace "prefix.map" to the actual map file
```

The latter command generates the `.txt` and `.html` reports.
