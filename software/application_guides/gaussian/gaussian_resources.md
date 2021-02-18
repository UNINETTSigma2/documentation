# Memory and number of cores

This page contains info about special features related to the Gaussian install made on Sigma2 machines, but also general issues related to Gaussian only vaguely documented elsewhere.

## Gaussian over Infiniband
First note that the Gaussian installation on Sigma2 machines, are currently Linda parallel versions, so they scale somewhat initially. On top of this, our Gaussian is installed with a little trick, where the loading of the executable is intercepted before launch, and an alternative socket library is loaded. This enables running Gaussian natively on the Infiniband network giving us two advantages:

* The parallel fraction of the code scales to more cores.
* The shared memory performance is significantly enhanced (small scale performance).

But since we do this trick, we are greatly depending on altering the specific node address into the input file: To run gaussian in parallel requires the additional keywords `%LindaWorkers` and `%NProcshared` in the Link 0 part of the input file. This is taken care of by a wrapper script (***gXX.ib***) around the original binary in each individual version folder. This is also commented in the [jobscript example](gaussian_job_example.md). Please use this example(s) as starting point(s) when submitting jobs.

Syntax is shown here:


	g16.ib $input.com > g16_$input.out

We have also taken care of the rsh/ssh setup in our installation procedure, to avoid `.tsnet.config` dependency for users.

## Parallel scaling
Gaussian is a rather large program system with a range of different binaries, and users need to verify whether the functionality they use is parallelized and how it scales.

Due to the preload Infiniband trick, we have a somewhat more generous policy when it comes to allocating cores/nodes to Gaussian jobs but before computing a table of different functionals and molecules, we strongly advice users to first study the scaling of the code for a representative system.

Please do not reuse scripts inherited from others without studying the performance and scaling of your job. If you need assistance with this, please contact user support!

We do advice people to use up to 256 cores (`--tasks`). We have observed acceptable scaling of the current Gaussian install beyond this count of cores/nodes for the jobs that do scale outside of one node (i.e. the binaries in the `$gXXroot/linda-exe` folder). But, Linda networking overhead seems to hit hard around this amount of cores, causing us to be somewhat reluctant to advice going beyond this until further notice.

Gaussian takes care of memory allocation internally. This means that if the submitted job needs more memory per core than what is in average available on the node, it will automatically scale down the number of cores to mirror the need. This also means that you always should ask for full nodes when submitting Gaussian jobs! This is taken care of by the `--exclusive`in flag SLURM, and also commented in the job script example.

**The `%mem` allocation of memory in the Gaussian input file means two things:**

* In general it means memory/node – for share between nprocshared, and additional to the memory allocated per process. This is also documented by Gaussian.
* For the main process/node it also represents the network buffer allocated by Linda since the main Gaussian process takes a part and Linda communication process takes a part equally sized – thus you should never ask for more than half of the physical memory on the nodes, unless they have swap space available - which you never should assume.

Please consider the memory size in your input if jobs fail. Our job example is set up with 500MB (which is actually a bit on the small side), test-jobs were ran with 2000MB. Memory demand also increases with an increasing number of cores, for jobs with 16 nodes or more - doubling the amount of `%mem` would be advicable. But this would also limit the size of problems possible to run at a certain number of cores.

<span style="color:violet"> To top this, the `%mem` tag is also influencing on performance; too high makes the job go slower, too low makes the job fail.</span>

The maximum `%mem` limit will always be half of the physical memory pool given on a node. As a rule of thumb, you should also leave a small part for the system. That is why we would actually advise to use not more than 15GB as maximum `%mem` size. **Note that "heap size" is also eating from the memory pool of the node (see below).**

## Managment of large files

As commented in the [performance-tips section](/files_storage/performance/lustre.md), there is an issue with very large temporary output files (termed RW files in Gaussian). It is advisable to slice them into smaller parts using the `lfs setstripe` command.

Note the placement of this command in the [jobscript example](gaussian_job_example.md). It is paramount that this commands comes after defining that variable and creating the corresponding folder.

## Important aspects of Gaussian setup on Fram:

On Fram, we have not allocated swap space on the nodes, meaning that the heap size for the linda processes in Gaussian is very important for making parallell jobs run. The line

	export GAUSS_LFLAGS2="--LindaOptions -s 20000000"

contains info about the heap size for the linda communication of Gaussian. <span style="color:violet"> 20 GB (the number above) is sufficient for most calculations, but if you plan to run 16 nodes or more, you may need to increase this number to at least 30 GB.</span>

On Fram, the gXX.ib wrapper does two things: Together with introducing the  explicit IB network adresses into input file, it distributes the jobs onto two linda processes per node and halves the processes per linda compared to processes per node. After thorough testing, we would generally advice users to run 4-8 nodes with heap-size 20 GB (see above) and 2 Linda instances per node for running Gaussian on Fram.
