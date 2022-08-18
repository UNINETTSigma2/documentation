---
orphan: true
---

# Memory and number of cores

This page contains info about special features related to the Gaussian install
made on NRIS machines, but also general issues related to Gaussian only
vaguely documented elsewhere.


## Gaussian over Infiniband

First note that the installed Gaussian suites are currently
Linda parallel versions, so they scale out of single nodes. In additidon, our
Gaussian installation is done with a little trick, where loading of the executable
is intercepted before launch and an alternative socket library is loaded. We have also taken care of the rsh/ssh setup in our installation procedure, to avoid `.tsnet.config` dependency on user level. This
enables Gaussian to run on Infiniband network with native IB protocol, giving us two
advantages:

* The parallel fraction of the code scales to more cores.
* The shared memory performance is significantly enhanced (small scale performance).

To run Gaussian in parallel on two or more nodes requires the
additional keywords `%LindaWorkers` and `%NProcshared` in the Link 0 part of
the input file. In addition, since we do the interception-trick, we need to add the specific IB network node address into the input file. This is taken care of by a wrapper script (`g16.ib`) around
the original binary in each individual version folder. Please use this example(s) as starting point(s) when submitting jobs.

Advised job submit syntax using the wrapper:

	g16.ib $input.com > g16_$input.out

## Parallel scaling

Gaussian is a rather large program system with a range of different binaries,
and users need to verify whether the functionality they use is parallelized and
how it scales both in terms of core and memory utilization.

### Core utilization

Due to the preload Infiniband trick, we have a somewhat more generous policy
when it comes to allocating cores/nodes to Gaussian jobs. 

- We strongly advice users to first
study the scaling of the code for a representative system. 
- **Please do not reuse scripts inherited from others without studying the
performance and scaling of your job**. We recommend to take our
{ref}`gaussian-job-examples` as a starting point.

Due to its versatility, it is hard to be general about Gaussian binary scaling. We do know that plain DFT-jobs with rather non-complicated molecules like caffeine scales easily up to the core-count limit on Saga and into the range of 16 nodes on Fram. On the other side, jobs with transition-metal containing molecules like Fe-corroles scales moderately outside of 2 full nodes on Saga. On a general note, the range on 2-4 nodes seems to be within decent scaling behaviour for most linda-executables (the LXYZ.exel-binaries, see the Gaussian home folder on each NRIS machine). Also note that due to different node-sharing policies for Fram and Saga there will be the need for setting an additonal slurm-flag when running Gaussian jobs on Saga.  

### Memory utilization:

The `%mem` allocation of memory in the Gaussian input file means two things:

- In general, it means memory/node – for share between `nprocshared`, and
  additional to the memory allocated per process. This is also documented by
  Gaussian.
- For parallel jobs it also means the memory allocation hte LindaWorker will make, but from Linda9 and onwards the need for a memory allocation on the master node has been removed. 

However, setting %mem to a value of less than 80% of the physical memory (the actual number depends on the actual node since we have standard-, medium-memory-, and big-memory nodes, see {ref}`fram` and {ref}`saga`) is good practice since other system buffers, disk I/O and others can avoid having the system swap more than necessary. This is especially true for post-SCF calculations. To top this, the `%mem` tag is also influencing on performance; too high makes the job go slower, too low makes the job
fail.

Please consider the memory size in your input if jobs fail. Our job example is
set up with 500MB (which is actually a bit on the small side), test-jobs were
ran with 2000MB. Memory demand also increases with an increasing number of
cores. But this would also limit the size of problems possible to run at a
certain number of cores. For a DFT calculation with 500-1500 basis sets, `%mem=25GB`
can be a reasonable setup. 

**Note that "heap size" is also eating from the memory
pool of the node (see below).**


## Management of large files


As commented in the {ref}`storage-performance`,
there is an issue with very
large temporary output files (termed RW files in Gaussian). It is advisable to
slice them into smaller parts using the `lfs setstripe` command.

### On Saga:
The corresponding situtation for Saga is described here: [About storage performance on Saga](/files_storage/performance/beegfs.md).

## Important aspects of Gaussian NRIS setup


### On Fram:

On Fram, we have not allocated swap space on the nodes, meaning that the heap
size for the Linda processes in Gaussian is very important for making parallel
jobs run. The line

	export GAUSS_LFLAGS2="--LindaOptions -s 20000000"

contains info about the heap size for the Linda communication of Gaussian
and the setting 20 GB (the number above) is sufficient for most calculations. This is the double of the standard amount for Gaussian, but after testing this seems necessary when allocating more than 4 nodes (more than 4 Linda-workers) and sufficient up to 8 nodes. Above 8 nodes, Linda-communication seems to need 30 - which amounts to half of the standard node physical memory and reducing the available amount for `%mem`accordingly. 