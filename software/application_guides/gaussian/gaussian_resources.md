---
orphan: true
---

# Memory and number of cores

This page contains info about special features related to the Gaussian install
made on Sigma2 machines, but also general issues related to Gaussian only
vaguely documented elsewhere.


## Gaussian over Infiniband

First note that the Gaussian installation on Sigma2 machines, are currently
Linda parallel versions, so they scale somewhat initially. On top of this, our
Gaussian is installed with a little trick, where the loading of the executable
is intercepted before launch, and an alternative socket library is loaded. This
enables running Gaussian natively on the Infiniband network giving us two
advantages:

* The parallel fraction of the code scales to more cores.
* The shared memory performance is significantly enhanced (small scale performance).

But since we do this trick, we are greatly depending on altering the specific
node address into the input file: To run Gaussian in parallel requires the
additional keywords `%LindaWorkers` and `%NProcshared` in the Link 0 part of
the input file. This is taken care of by a wrapper script (`g16.ib`) around
the original binary in each individual version folder.
Please use this example(s)
as starting point(s) when submitting jobs.

Syntax is shown here:

	g16.ib $input.com > g16_$input.out

We have also taken care of the rsh/ssh setup in our installation procedure, to avoid `.tsnet.config` dependency for users.


## Parallel scaling

Gaussian is a rather large program system with a range of different binaries,
and users need to verify whether the functionality they use is parallelized and
how it scales.

Due to the preload Infiniband trick, we have a somewhat more generous policy
when it comes to allocating cores/nodes to Gaussian jobs but
we strongly advice users to first
study the scaling of the code for a representative system by running it
for instance on 16, 32, 64, ... cores (tasks) to observe for which number of
cores (tasks) the scaling saturates.

Please do not reuse scripts inherited from others without studying the
performance and scaling of your job. We recommend to take our
{ref}`gaussian-job-examples` as a starting point.

The `%mem` allocation of memory in the Gaussian input file means two things:

- In general, it means memory/node – for share between `nprocshared`, and
  additional to the memory allocated per process. This is also documented by
  Gaussian.
- For the main process/node it also represents the network buffer allocated by
  Linda since the main Gaussian process takes a part and Linda communication
  process takes a part equally sized – thus you should not ask for more than
  half of the physical memory on the nodes (the actual number depends on the actual node
  since we have standard-, medium-memory-, and big-memory nodes, see {ref}`fram` and {ref}`saga`).

Please consider the memory size in your input if jobs fail. Our job example is
set up with 500MB (which is actually a bit on the small side), test-jobs were
ran with 2000MB. Memory demand also increases with an increasing number of
cores. But this would also limit the size of problems possible to run at a
certain number of cores. For a DFT calculation with 500-1500 basis sets, `%mem=25GB`
can be a reasonable setup.

<span style="color:violet"> To top this, the `%mem` tag is also influencing on
performance; too high makes the job go slower, too low makes the job
fail.</span>

The maximum `%mem` limit will always be half of the physical memory pool given
on a node. As a rule of thumb, you should also leave a small part for the
system. That is why we advice to set `%mem` no higher than half
the available physical memory. Note that "heap size" is also eating from the memory
pool of the node (see below).


## Management of large files on Fram

As commented in the {ref}`storage-performance`,
there is an issue with very
large temporary output files (termed RW files in Gaussian). It is advisable to
slice them into smaller parts using the `lfs setstripe` command.


## Important aspects of Gaussian setup on Fram:

On Fram, we have not allocated swap space on the nodes, meaning that the heap
size for the Linda processes in Gaussian is very important for making parallel
jobs run. The line

	export GAUSS_LFLAGS2="--LindaOptions -s 20000000"

contains info about the heap size for the Linda communication of Gaussian
and the setting 20 GB (the number above) is sufficient for most calculations.

On Fram, the g16.ib wrapper does two things: Together with introducing the
explicit IB network addresses into input file, it distributes the jobs onto two
Linda processes per node and halves the processes per Linda compared to
processes per node. After thorough testing, we would generally advice users to
run 4-8 nodes with heap-size 20 GB (see above).
