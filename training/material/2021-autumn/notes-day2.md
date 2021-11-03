---
orphan: true
---

(training-2021-autumn-notes-day2)=

# Questions, answers, and notes from day 2


## Ice breaker

Have you ever used any of these?

(you can vote by adding an "o" at the end)

- MPI              : ooooooooo
- OpenMP           : oooooooo
- `multiprocessing`: ooooooo
- CUDA             : ooo
- OpenACC          : oo
- `rayon`          : oo
- TBB              :
- SYCL             : o
- `numba`          : ooo
- (what else is there?)


## Links

- Session 1 [Slides](https://drive.google.com/file/d/1jSfdYK0WdHfMYICS-AZ30qsW8yp5MYpg/view?usp=sharing)
- Session 2 [Slides](https://drive.google.com/file/d/1jy6wvidIuHb8aWpA1pN0qd7ULm7ta4Yt/view?usp=sharing)
- Session 3 and 4: [Slides](https://cicero.xyz/v3/remark/0.14.0/github.com/bast/talk-better-resource-usage/main/talk.md/)
  - [Tutorial about memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)
  - [Tutorial about cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)
- Session 5 : [Slides](https://drive.google.com/file/d/1NGiXFZQvVlDXCJhxBJyMgogegOGl7BOG/view?usp=sharing)


## Questions and answers

- How can i run more than one core in SAGA?
    - Use the `--cpus-per-task=X` Slurm flag for threads (e.g. for OpenMP) or `--ntasks=Y` if you would like independent cores (e.g. for MPI ranks)
      - [later today](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html#what-is-mpi-and-openmp-and-how-can-i-tell) I will also demonstrate how you can find out whether this is an OpenMP-aware code and/or MPI-aware code (it can be difficult to tell)

- How to decide if my run is too large for the `$home` directory?
    - Two things size of the output and number of files. Output may not be just final one, but also temporary files created during the run. Use the `dusage` command to find out current usage and your limit. Then decide if to use the `$HOME` directory or not. In general, as a personal preference I do not use `$HOME` to place output from jobs, I use the workfolder `/cluster/work/users/$USER` , then move the results I want to keep to one of the project folders. Please note `/cluster/work/users/$USER` is temporary and files will be deleted after sometime (not a place to keep things you want to keep)
    - can `/cluster/work/users/$USER` be used to develop code as well?
        - Yes, please note it is a temporary location (a scratch)
            - [More details 1](https://documentation.sigma2.no/jobs/job_scripts/work_directory.html)
            - [More details 2](https://documentation.sigma2.no/files_storage/clusters.html)
        - I use git for version control anyway, so I can always clone it, no?
            - For code I do not think that is a good idea. The `/cluster/work/users/$USER`  is better used for large data scratch (large files, and large number of files)
        - So where should I develop my code?
            - I would say your project folder. :+1:
- What is difference between MPI and OpenMP?
    - MPI can accommodate multiple nodes in addition to use multiple cores on one node. OpenMP is to run parallel on a single node (please see the hybrid question below)
    - To add to the confusion, please note OpenMPI is a MPI version
    - [Here is a good article](http://pawangh.blogspot.com/2014/05/mpi-vs-openmp.html)

- What is the difference between node and thread?
    - A node is a separate server, a thread is a virtual core on a CPU
    - So a server/node can have for example 32 cores but 64 virtual cores, aka threads
    - Threading is a way to increase the performance of a CPU without adding more physical cores

- can you explain what hybrid means in hybrid job
    - A hybrid job is a job that combines MPI with threading (e.g. OpenMP)
    - A hybrid job asks for multiple cores per task (`--cpus-per-task`) and multiple tasks (`--ntasks=X` and/or `--ntasks-per-node`)
    - Would something like Posix threads be used instead og OpenMP?
        - Yes! That is why we use words such as "threading" instead of just writing OpenMP, there are many ways to utilize these resources, examples are:
            - OpenMP threading
            - Posix threading
            - Intel Thread Building Blocks (TBB)
            - `std::thread` in C++11

 - can you give an image for explaining the difference between MPI ranks and threads?
     - Not something we made, but - [Here is a good article](http://pawangh.blogspot.com/2014/05/mpi-vs-openmp.html)
     - This link might be useful: <https://hpc.llnl.gov/training/tutorials/introduction-parallel-computing-tutorial#ModelsOverview>

- Is the slurm command
    $SBATCH --ntasks
    (without-per-node) sometimes useful? What happens if --ntasks-per-node     is also set?
    - They are both very useful and your example is excellent for giving a good homework task
    - When combining `--ntasks` with `--ntasks-per-node` you will get a total of `--ntasks`, but on any given node there will only ever be `--ntasks-per-node` tasks per node
        - Which means you will get `--ntasks` / `--ntasks-per-node` nodes
    - For what it's worth, I [name=radovan] most of the time only use `--ntasks` on Saga and I let the machine decide where to put them. This minimizes queuing time. For many codes this is perfectly fine. Some codes need to have tasks distributed in a specific way and then I go for the more rigid `--ntasks-per-node`. On Fram and Betzy I am trying to use full nodes.

- Does Betzy have NVMe local scratch?
    - Unfortunately not

- What is Scratch?
    - Scratch is a common name used for a place (directory) to store temporary results
    - Within Slurm, a variable `$SCRATCH` is set to point to a directory where your job can store temporary data
        - This data is removed after the job is finishes so you need to be careful to copy data you would like to keep to another storage area
        - [More information can be found here](https://documentation.sigma2.no/files_storage/clusters.html#job-scratch-area)
    - How about Local Scratch?
        - Local scratch is simply a scratch directory that is local to the node(s) that you work on. Local scratch is usually a lot faster than "normal" scratch, but it can be more difficult to use since it is local to each node

- Should I store data at Scratch/LocalScratch and then copy it to lets say to `$USERWORK` or save the data directly in `$USERWORK`?
    - Scratch and LocalScratch are two defferent thing. LocalScratch is on the computer node it self and not network mounted, which has the fastest IO (read/write). `$USERWORK` and Scratch no performance difference, but can access from multiple nodes.
    - Then `$USERWORK` is temporary as well

- Take the CPUs on Betzy as an example: Whats the point of having two threads (correct terminology?) per physical core (256 threads, 128 cores per node) when we only should use the number of physical cores? The doc. says that there is no point in using all 256 threads as it has no performance benefit over using 128.
    - On Betzy and other systems only physical cores are used. When programs try to use threads it would use pysical cores.
    - [Betzy efficient usage](https://documentation.sigma2.no/jobs/memory-bandwidth.html)
    - The reason for this recommendation is that SMT cores can't run at the same time, but instead run concurrently, which means that one thread runs while the other is stalled. For scientific workloads a stall usually means that the thread is waiting for memory and not IO and here we run into the memory bandwidth barrier that Ole talked about (the other SMT core can't do anything useful because it would likely also wait for memory)
        - Then why are ~~multithreads~~ SMT ever useful if they cannot work at the same time?
            - On your laptop it can be useful because most of the time is spent waiting for IO (either from disk or from the network) so here SMT cores makes sense, because two independent programs can "run at the same time" on one core because often one program is waiting, then we run the other and vice versa
                - Thanks!
            - Also note that SMT is not the same as multithreading, SMT is where one core "behaves" as two independent cores (but in reality it is not actually independent)
                - Allright, good to have the terminology sorted out!

- what is the difference between OpenMP and MPI?
    - The exact question aswered above.
    - but in short: OpenMP communicating through memory, only on 1 node. MPI: communicating via messages, can go to many nodes.

- How we can use shared memorey in our simulations?
    - You will have to use some form of threading, e.g. OpenMP to have multiple threads work together on the same data
    - If you can comment in which language you work we can point you towards more documentation
        - I am using FLUENT and REEF3D which is an open source package written in C++
            - FLUENT should be able to simply pick up the correct number of threads to utilize
            - Try a small test job and ask for `--ntasks=1` and `--cpus-per-task=40` on Saga it should be able to use all the cores
            - If this doesn't work, send us an email at support@nris.no and we can help you further

- Can you please clarify how ranks, tasks, threads, cores and CPUs relate to each other?
    - We will try, but note that depending on the context these terms might be used interchangeably
    - A CPU is the processor that perform our calculations. A CPU _can_ contain multiple cores. When we run a program it starts one thread which is the "thread" of execution for this program, this thread can "split" out into multiple threads to share the work.
    - Then we move over to Slurm, in Slurm we use task to talk about running a single program. This Slurm task can then use multiple threads (`--cpus-per-task`), but we can also have multiple tasks that work together (`--ntasks`).
    - Then we have MPI terminology, within MPI we call the Slurm task a _rank_.

- How many threads should we set per each task/cpu? what is the optimum amount?
    - It is diffcult to give a number for this due to following reasons:
        - Number of cores can be different even on the nodes of the same system
        - Memory access, how much memory each thread needs. [Betzy, NUMA](https://documentation.sigma2.no/jobs/memory-bandwidth.html)
            - Can the local memory banks be used (Refer the NUMA article above)
        - The software, can the software exapand and use everything you through at it
        - I would say the best is to test with smaller sample/iteration and plot and undestand the trend (as Ole is showing)

- Does it has a same function of srun and mpirun for MPI run in SBATCH?
   - do you mean whether both srun and mpirun can be use interchangeably in job scripts?
   - It is meaning whether they are equally used in SBATCH script?
       - Use `srun` on Saga and fram and `mpirun` on betzy. [Why this is?](https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html)
       - Thanks. In fact, on Betzy srun also works like:   srun --mpi=pmi2 -n $NMPI ./hycom_cice. So it is why I feel some confusing for them.
           - Thank you, we shall have a discussion today latter and try to find a best way from the usability point (what is least confusing for user).

- comment on `htop`
  - w**onderful tool, available on Betzy and Fram. bu**t on Saga it's only on login nodes. on compute nodes on Saga you have to use the older and less shiny `top`.
    - question to colleagues: any good reason not to install htop also on Saga compute nodes?
        - No one asked yet, now you have so I think we will have it soon.

- Are there any good resources on how to set up a cluster of Raspberry Pis to behave like the structure of, for example, Betzy? For running hybrid jobs.
    -  There is to my knowledge no code available which will replicate Betzy on any other system
    -  We have automated the setup to a large extent but I'm sure there are security considerations before we can make (part of?) the code public
    -  But long term our plan at the scientific computing group at the IT department in Bergen is to offer our small, 4 node, Raspberry Pi cluster as a service of some sort
    -  The details haven't been ironed out but it's in progress
        -  Thanks for the info!

- can you explain again the difference between sequential and random access to memory (SAM vs RAM) and what is the standard on saga/fram/betzy?
    - Note that there is a difference between _how_ we access memory, either sequential or random and the _type_ of memory
        - RAM is a _type_ of memory (type of hardware) which is more or less equally fast to access any bank within the hardware (hence the Random in the name)
        - When we access memory we usually access it either sequentially or random. When reading a large file it is stored one bit after another and so we read the large file sequentially. However, when we want to write small files e.g. we create `result_1.txt` with result after one time step and then create `result_2.txt` with the result after two time steps these files are written more or less randomly (the Operating System, can decide to place the bits of these files far away from each other).
        - For hard disk sequential read is preferred because the read head does not need to move much when reading. Most of the storage on Saga, Fram and Betzy consist of hard disk so sequential read is quite a bit faster than random

- is there documentation on the amount of memory limits per core on saga?
    - The technical documentation for Saga [can be found here](https://documentation.sigma2.no/hpc_machines/saga.html#main-components)
    - As can be seen Saga has 8 "big mem" nodes with 384 GiB of memory so the theoretical maximum there is 384 GiB all on one core or `384 / 40 ≃ 9GiB` per core

- what's the difference between memory (mem) and virtual memory (vmem) when looking at a job ressource report?
    - Are you referring to the log file created at the end of the job ?
    - yes, the output file (or when checking job stats after it has ended e.g. I was used to use qstat jobid)
        - Description of each field is [here](https://slurm.schedmd.com/sacct.html), I see two virtual memory realted fields there (MaxVMSize and AveVMSize).
        - These refers to RAM usage and not SWAP (on disks) if that is what you were wondering
        - If you can find the exact variable name (e.g. AveVMSize) we can try to find more details.
        - I think I meant MaxVMSize, why is it "Virtual Mem"
            - MaxVMSize represents the maximum amount of physical memory used by all processes in the job. On our system this is always RAM. The name is a general SLURM variable name. This comprises all memory forms involved.

- I often use `ipython` to run quick interactive sessions on the login nodes to do some simple checks on the data generated from my jobs. `ipython` is terribly slow to start up, like 20 seconds before I can type anything. Do you know why?
    - Most likely this is due to the parallel filesystem which is not optimized for such programs. I would hazard a guess that `ipython` tries to read many small configuration files when it is starting which result in the slow start-up
        - If I close and quickly re-poen `ipython`, it usually starts up very quickly.
            - Most likely because the files that `ipython` just read was cached, but this is more of a guess from my side
                - Sounds like a reasonable guess! Thanks!

- Is the heat from any of the supercomputers used for anything? Like from the watercooling of Betzy.
    - The return-water from Betzy is used to heat up campus at NTNU Gløshaugen.
    - ~~Not at the moment, but~~ that is something very relevant for the next system i Norway
        - Info on the next system?
            - Not yet :innocent:
                - :disappointed_relieved:

- When will e-mail alerts be back?
    - Unfortunately there is no fix for this issue and no ETA for a fix either
        - What is really the problem? In my naive mind, I think it sounds very easy to send a few e-mails!
            - The main challenge is that people tend to write the wrong email address which result in a lot of noise when emails bounces and end up in our ticketing system
                - We got > 200 emails per day for this because people start array jobs where each step result in an email that could possibly fail
                    - Too bad... I really liked the feature! :worried:
                        - Us as well which is why we want to solve this, but it turned out to no be easy to set this up correctly

- Consider this scenarion (true story): users create jobscripts and configurations to run their jobs. Benchmark is done, and an optimal configuration is found. Then comes cluster downtime, with update/upgrade of the machine. The results from the banchmark are then not anymore reproducible. Users then need to rerun banchmark and reconfigure jobscripts. How to deal with this more efficiently? I mean,  rerunning benckmark jobs takes resources and time, so we (users) tend to not do it. Perhaps, some benckmark should be centralized to avoid multiple users to runs very similar banchmarks. Is there any such "centralized banchmark" on NRIS machines?
    - Example: After the upgrade the performance might not be optimal anymore. Certain jobs that could be run well with the old (pre-upgrade) configuration using N resources and took M runtime, on the new configuration (post-upgrade) take more runtime (with N resources), or require more than N resources (i.e., larger memory footprint).
    - We shall alocate some time for you to raise question later on, as there might be some thing we need to chnage from our policy side.
        - :+1: (NB: I'm following from the airport, and soon I'll be boarding, so I cannot participate "live", sorry).
            - OK, we will archive this with the answer, please check back later
                - :+1:

- Radovan just said "you asked for 16 cores, but there are atually 20 cores, so 4 are wasted". Could anyone elaborate on this? Why 4 are wasted? Won't them be used by other users?
    - If you use only 16 cores, then you have to pay for 20 cores.
    - That is true, the user requesting 16 cores are not wasting the 4 remaining cores, but they are not using the resources effectively either
    - If everyone asked for only 16 cores then finding work to place on these 4 cores could be really difficult resulting in longer queuing times
    - On Saga: no problem, you can ask for as many cores as you like and the remaining cores can be used by others.
    - On Fram/Betzy: jobs are allocated as _full_ nodes, so here remaining cores will actually be wasted
    - [name=Radovan] sorry I was imprecise when speaking but what really
      happened is that people asked for an exclusive node and 16 tasks per node
      so in that case the 4 processors possibly were wasted.

- So for Fram, you should always ask for 32 cores if you specify cores per node?
    - On FRAM we ask for a whole nodes (inbstead of cores by number), [more details](https://documentation.sigma2.no/jobs/job_types/fram_job_types.html)
    - so e.g: --ntasks=256 --ntasks-per-node=16 will automatically assign 16 nodes using all 32 cores each?
    -Yes this is correct for ntasks=256
    - Sorry I am confused again... if the node always uses (or at least reserves) all 32 cores on the node, what does the --ntasks-per-node really do? Why isn't there always 32 tasks-per-node and 8 nodes instead? (in the example of --ntasks=256)
      - It is a good question. If we always allocate full nodes, we could simply divide the number of tasks by the number of cores per node. But some codes
        need a smaller --ntasks-per-node to "stretch" to have access to more memory.

- could you give examples for queing times for different sized jobs? e.g. 1 cpu - low mem jobs vs. 40 cpu - full avail mem jobs per node (i know this depends on the workload, just some experience)
    - User experience: I got 128 nodes for 2 days on Betzy with 0 queue time a few days ago. But a few days before that, I had to wait for two days to get 128 nodes for a few hours. I guess the queue time depends on many factors!
    - The queue time is dependent on the current amount of jobs, so it may be 1 second or several days no matter the amount of resources you are asking for
    - Generally jobs requesting few resources will start sooner than the opposite kind
    - Tip for lower queing time on Saga: use the general `--ntasks=(N*n)` instead of specifying `--nodes=N --ntasks-per-node=n`. It will give the scheduler much more freedom for fitting your job. In particular, don't ask for `ntasks-per-node=40` i.e. _full_ nodes on Saga (unless you really need it), such jobs tend to queue for a _long_ time.

- Do you have an overview of recommended compiler optimization flags to try?
    - You can find [our recommendations here](https://documentation.sigma2.no/computing/tuning-applications.html)
        - Hm, I cant find any compiler flags in that article... Or am I mistaken?
            - You are right, a bit to quick on the draw, [this was the documentation I wanted to link](https://documentation.sigma2.no/code_development/building.html)

- is there a limit on jobs a user can submit to the queue under 'normal' at a given time? (i have run into a limit under 'devel'). example: workflow managers (like snakemake or nextflow) submit a lot of computational easy jobs in parallel.
    - Yes there are limit. [Details](https://documentation.sigma2.no/jobs/choosing_job_types.html)
        - it says for SAGA-normal: Job limits 1-256 units where unit is # of cpu for low mem jobs. so in a workflow manager i would limit the number of parallel submitted jobs to sth lower than 256. so this, in fact, limits my parallelization.

- In my script, there is no of nodes and --ntasks-per-node, but not no of cores. Is it good or bad?
    - --ntasks-per-node would refer to cores. i.e. I need this many cores on easch nodes
    - It will depend on your application, if it is pure MPI the only specifyling `tasks` is correct, then you get the default `--cpus-per-task=1`. If your application is using shared memory OpenMP or hybrid MPI+OpenMP, then you should also use the `--cpus-per-task` keyword.
    - Clarifying cores vs tasks vs `cpus-per-task`:
        - if `ntasks=N` and `cpus-per-task=n` then "number of cores requested" = N*n

- About test datasets. My programs need to detect a pattern in the whole dataset (genome), and then it get error because the truncated version is "not complete". How would you approach this? Maybe I misunderstand my own problem.
    - For this specific example, not truncate but partitioning is the solution. i.e. finde the fastq boundary and split.
    - Or you could setup the test using a smaller genome.

- If I ask for 5 minutes and 16 CPU (or lots of memory), will I get charged for the memory, even if the job crashes after 5 minutes?
    - Yes, for the 5min the job ran
    - But the 5 min until the crash resources will be reserved (for the whole resource request).

- Core VS node. Are nodes a group of cores? are nodes "the restaurants" and cores "the chairs"?
    - Restaurant: full cluster (Saga, Betzy, Fram)
    - Table: compute node (typically c1-23 etc)
    - Chair: CPU core/thread
    - Butler: Resource manager/scheduler (Slurm, PBS, etc)

- is there any memory overload that is NOT tracked in "top" or in the slurm log?
    - Radovan just answered: sampling could be too slow to detect the peak.
    - Slurm samples 30 seconds intervals. top perhaps every second? but disadvantage of using top is that you actually have to watch it. it's nice to get a summary after the calculation.

- Maximum resident set size. What is this exactly?
    - This is the largest amount of memory that Slurm actually detected your program using and interacting with
        - It comes back to the difference between virtual/actual memory

- how fine-grained does the memory requirement estimation should be? does it matter if i ask for 100M vs 500M vs 1G vs 2G vs 12G per CPU?
    - so it is total mem per node/ number of cores in this node. thats the mean mem per cpu?
    - A few hundred MB to 1GB for the total job is nothing to worry about, but if it is per core then you should not oversubscribe by more than a few hundred MBs

- This --mem-per-cpu was specific for Saga, right? So I assume we can check the memory needed in the same way in e.g. fram, but if --mem-per-cpu is Saga-specific, how is it then assigned in Fram? or is is not?
    - On Fram for example as you ask for the whole node, all memory attached to the node are at your disposal.
    - On SAGA we can ask for less than a full node, that is why memory is a parameter
    - Ah, I see. So for doing the same thing on Fram, what we need to do is to adjust the number of tasks and tasks-per-node that we ask for until we land on an appropriate memory usage?

- is the monitoring of memory with `time` reliable for executables running in parallel (multi-node)? This is non-mpi stuff (Linda-parallelized Gaussian).
    - You could also get similar info from SLURM (try the following for the job Radovan ran)
        - sacct -j  4318034 -o AveCPU,AvePages,AveRSS,AveVMSize,MaxVMSizeNode,MaxVMSizeTask
            - uhm... results in an empty table (on saga, with my job number, of course)
            - May the job number not exact match (please share if it is OK)
                - 4305243 - Reason is, this is still running, the above summary available only at the end
                    - ah... OK, thanks

- The MaxRSS reported in slurm log is the sum of all cores on all nodes allocated for the job, is that correct ?
    - That is the *Maximum resident set size of all tasks in job.* i.e. the max reported if more than one nodes reported values.
        - so if a job uses 2 nodes, one node MaxRSS is 4G and the other is 6G, then the log reports MaxRSS is max(4,6)=6G?

- can you show again what was your run.sh script? with respect to example.f90
  - <https://documentation.sigma2.no/jobs/choosing-memory-settings.html#by-checking-the-slurm-output-generated-with-your-job> (but I removed the compilation part)
  - ah ok, that's whyI didn't see it.
  - what about running gfortran example.f90 -o mybinary as is. It creates the mybinary output... what is this?     -
    - yes that line compiles example.f90 into mybinary (this step is not so much relevant to this topic so I didn't want to go into this. you can leave it in the script also since it just takes a second)

- the mentioned 800GB $LOCALSCRATCH ... is an NVME on each Pizzabox/Node or is it shared among nodes?
    - This is local to each node (or Pizza box), so each pizza box has it's own [little stick of storage](https://www.servethehome.com/wp-content/uploads/2020/08/SNV3400-400G.jpg)

- What is the link to the tutorial about optimization of jobs and memory? The demo with `usr/bin/time -v`
  - [Tutorial about memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)
  - [Tutorial about cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)

- How do you know the number of tasks?
    - We start with a random "small" number of tasks to just test, could start with 1, 2, 4, 8, it is a bit arbitrary to begin with

- In general, is it better to increase the number of cores instead of memory per core? And what is the maximum memory per core on Saga?
    - The maximum memory per core is limited by the total amount of memory available on the node, you could ask for all the memory (196GiB for normal nodes on Saga) for one task or ask for `196 / 40 ≃ 5GiB` per core if you would like to use many cores
    - Is it better to increase cores or memory?
        - As long as your job has a enough memory, asking for more does not make it faster
        - However, asking for more cores could make your program faster
        - So once you have found the correct amount of memory, try to increase the number of cores as explained until you find the "optimal" number of cores for your program
        - And for Fram, figure out how much memory is needed and the optimize the nr of tasks and tasks per node?

- Question from Oles presentation. He spoke about reserving 64 cores, not 128, from each CPU on each node of Betzy. He then showed a nice htop where cores 1 to 128 were in use (all cores on the left two columns in htop), easy to see that exactly 64 cores are used from both CPUs. If I let the OS allocate cores (specifying that I want only 128), how do I know that it uses exactly 64 cores from each processor and not, for example, 66 cores (using SMT here, which I do not want) from CPU1 and 62 cores from CPU2? My htop is not as symmetric as the one Ole showed!
    - And should I make the program use cores 1 to 128 specifically? Is that better than using 64 "random" cores from each CPU? If so, how do I tell my program what cores to use?
        - The default is normally OK, but if you really want to make sure you need to use variables like : export I_MPI_PIN_PROCESSOR_LIST=all:socket
       - Check the Intel software manual to learn more about this. Record run time between default and the different settings you use.
       - <https://www.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/process-pinning/environment-variables-for-process-pinning.html>

- Where do we find numpy?
    - On our clusters you should load a module, e.g. `module load SciPy-bundle/2021.05-foss-2021a`
    - Then import it in your scripts with [`import numpy as np`](https://numpy.org/)
    - Also the `module load Python/3.6.6-foss-2018b` in the example script should bring numpy into view.
        - It does!

- Some tools provide their own options to define the number of cores/threads, such as [samtools](http://www.htslib.org/doc/samtools.html) that provides this parameter "nthreads=INT Specifies the number of threads to use during encoding and/or decoding." How to cooperate it with the ncores defined by #SBATCH?
    - In this case what you ask in SBTACH and what you infom the tool (in this case samtools) should mathch.
    - Best way to do this is using SLURM variables, let me find a samtools example..
        - The Slurm variable [`SLURM_CPUS_PER_TASK`](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_CPUS_PER_TASK) or [`SLURM_NTASKS`](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_NTASKS) can be used in your Slurm script

- my genome assembler uses OpenMP. Is the environment variable `OMP_NUM_THREADS` always available on the compute nodes or do i need to export it.
    - It is always set by Slurm when you start a job with multiple cores per task, the variable `OMP_NUM_THREADS` is set by `CPUS_PER_TASK`, `#SBATCH --cpus-per-task=1`
        - Cool!
        - Hm! good to know! I should try to simplify the example on the web.

- Marcin says that in Fortran, allocate(...) does not actually allocate anything before "first touch". Is it the same for C / C++
    - Yes, when we allocate memory, with e.g. `calloc` or `malloc`, the Linux kernel will "reserve" the memory, but it will not load any memory, it is not until you actually start to "touch" the data that Linux will start to "give" you memory
    - A simple example is to `malloc(2 x amount of RAM)`, this will work even though you ask for more memory than available, but if you never touch this memory Linux does not have to actually reclaim memory to give to your program
        - And it is true also for `calloc` which zeros the memory in addition?
            - Yes, when you asked for this zeroed memory Linux will, as you try to touch this memory, zero it just before you get the memory
            - `calloc` in addition to being safer can often be quicker and for modern programming languages (like Rust) it is quite difficult to actually get non-zeroed memory
                - Huh, how can it be quicker to get zeroed memory than not zeroed memory?
                    - Linux actual has a pool of ready to go zeroed memory which it can give out without doing anything, but for non-zeroed memory it will often have to traverse the "free-list" to find a suitable chunk of memory
                        - Ahaa, enlightening! Thanks!

- How to reqeust an advanced user support
    - [Details](https://www.sigma2.no/advanced-user-support)
    - [See also here](https://documentation.sigma2.no/getting_help/extended_support.html)

- can you clarify the differences between _--ntasks-per-node_ _--ntasks_ parameters? and _--cpus-per-task_ is the ncpu you want to use per node, right?
    - `--ntasks`: give me a number of tasks, anywhere available, can be on same node but can be distributed across nodes (2 tasks here, 4 tasks there, 3 tasks over there)
    - `--ntasks-per-node` this is used when you want to control precisely how many tasks are on a node. some codes need this, it can have good effects on performance, but it will also restrict a bit the scheduler so you may queue longer. personally I only use it if I am sure it is better for the code.
    - `--cpus-per-task` this is used for OpenMP or hybrid calculations where you typically want to place tasks on different nodes, and then on each node you want to run a number of threads (I guess it should be called threads per task to be less confusing?)

- Wouldnt it be better to completely disable SMT if it is not usable for HPC? Will probably remove some confusion!
    - For Saga this is actually the case, but it is difficult to find a one-size fit all solution

- is a logical core the same as a physical core? in contrast to virtual cores/threads
    - Virtual core == logical core
    - A physical core is the actual hardware which can contain one or two (traditionally) logical/virtual cores

- Why does OpenMP default to the number of virtual cores, when HPC work does only benefit from using the number of physical cores (it at least does so on my Macbook!)?
    - Because everyone lies... Well, all the usual operating systems (e.g. Linux, MacOS and Windows) tells our code that these two virtual cores are fully functional "proper" cores, "trust me I know a good core for you to run on"
    - OpenMP could ignore this, but depending on the type of workload using two virtual cores can actually help with performance, but in HPC as we have talked about it doesn't
        - So OpenMP opts to not ignore the OS and annoy those of us working with HPC instead
            - :sob:

- This is similar to a line of questions above regarding Fram: Why would you write --ntasks=256 --ntasks-per-node=16 giving 16 nodes when Fram always uses all 32 cores? Why wouldn't you instead write --ntasks=256 --ntasks-per-node=32 giving 8 nodes? That's the main question, but an addition is: could you also include --cpu-per-task=2, which would result in 512 threads? AND, is it likely that this would be an advantage, to divide the cores into threads?
    - One reason could be memory, `--ntasks-per-node=16` would give you twice the amount of memory in total, compared to `--ntasks-per-node=32`
        - But it amounts to the same nr of tasks?
            - yes, total number of tasks is the same
        - Still confused, sorry...
            - take the extreme `--ntasks=256 --ntasks-per-node=1`, then you still get 256 MPI tasks in total, but each will run on a separate machine (node) and thus have access to the full memory on the node (64G on Fram), but of course then you occupy (and pay for) 256 full compute nodes for the calculation and 31 out of the 32 cores on each node are idling
        - Thank you, this was clarifying! Although only 1 of the cores are used, the full node memory is available. So --ntasks-per-node=16 is a middle ground between more memory without being to expensive/demanding
            - :+1:
    - Dividing tasks into threads is only beneficial if your code is "threaded", e.g. OpenMP parallelized


## Feedback for the day

One thing you particularly enjoyed
- Performance impact of different compiler optimization flags.
- Tutorial about memory/core by Radovan
- Useful info about monitoring performance during runs, and the option to log in to compute nodes and run e.g. "htop".
- This hackmd is super useful! :) +1
- agree, the hackmd is a great resource!
- personal favorite: lecture and tutorial by Radovan. super clear. +1

One thing we should remove/change/improve
- It would be useful with a heads up whenever something is Saga, Fram, Betzy specific. It's confusing to keep track of all the thread, core, node etc specifications when it's a bit different between the systems and we don't always know which specific system we are talking about.
- Explain better HOW to choose between mpi and omp when parametrizing jobs
- In the part about improving jobs scripts for better usage, it would have beebn good to explain how to determine(set) where output files and error files are produced when submitting a script. Overall, it would have been nice to have a little overview on HOW to organise yourself on a cluster: home space vs work space vs scratch (where to put scripts, where scripts outputs etc…)
