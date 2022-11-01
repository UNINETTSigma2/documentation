---
orphan: true
---

(training-2022-autumn-best-practices-notes-day1)=
# Questions and answers from Best Practices on NRIS Clusters Nov 2022, Day 1 (Nov 1)

```{contents} Table of Contents
```

## Icebreaker Question 

- What would you like to get out of this workshop?
    - Better scripting for repetitive tasks
    - Learning how to run code on clusters optimally, for example how to set the number of nodes, tasks, tasks per node, cpus per task, threads per core, etc. in a sensible way. 
    - Using VScode on server

## Questions 

### Hardware overview

1. Is there internet access from compute nodes? if not why
    - Not all sites are reachable from compute nodes.
    - In special cases it is possible to setup a proxy. 
    - Compute nodes are meant to be used in analysis work, waiting for data download might  waste resources.
       - RB: For very many use cases the occasional network access might not have any effect on resource use for others.
    - RB: as far as I know/understand, there is some network access from compute nodes via so-called proxy servers which allows access such as "git clone" and possibly "pip install" and access to license servers and similar. If your application needs network access and it cannot access it from compute, can you please reach out to us? It might be fixable.
    - http/https access usually works, e.g., downloading with `wget`, `curl` and using package managers. However, it works differently for interactive and batch jobs. For interactive jobs, the environment should be set up correctly. Test by submitting an interactive job and then run `env | grep http`. If you see two lines such as `http_proxy=http://proxy.saga:3128/` and `https_proxy=http://proxy.saga:3128/` the environment is set up correctly (note, `http` without the `s` for `https_proxy` is not a typo). If you run a batch job, you may have to add the parameter `--get-user-env` when submitting jobs (note, this parameter is disabled for interactive jobs). Again, in the job, one can check if the environment is set up correctly by running `env | grep http`. If you see something like `http_proxy=http://proxy.saga:3128/` and `https_proxy=http://proxy.saga:3128/` the environment is set up correctly (note, `http` without the `s` for `https_proxy` is not a typo).

2. Explanation: NVMe drives: fast solid-state drives which are very "close" to the cores and where file operations don't have to travel through network and do not affect the parallel file system. In other words if your application can use local disk, please do.
   - only available on Saga
      - or also Fram? "A job on Fram/Saga can request a scratch area on local disk" 
      - Yes, also on Fram.  On Saga, it is NVMe, while on Fram, it is SSD (also faster than /cluster, but not as fast as on NVMe).
   - more info: https://documentation.sigma2.no/files_storage/clusters.html#job-scratch-area-on-local-disk

3. How do I use this so called local scratch
     - We have documented the instructions [here](https://documentation.sigma2.no/jobs/scratchfiles.html)
     - also: https://documentation.sigma2.no/files_storage/clusters.html#job-scratch-area-on-local-disk

4. What is object storage? How does it differ from file storage?
    - As users you could use this as normal storage, i.e. no need to do anything different.
    - Usually you have to use specific tools to access object storage, that is for storing data, retrieving data, for listing what is stored in the storage. So it works differently than a normal filesystem (like `$HOME`). There may exist tools which let you mount a "bucket" into your file system and then you can use it like a normal file system. E.g., you could try that within a container. 
    - If you want more details, it is discribed [here](https://documentation.sigma2.no/files_storage/performance.html)

5. How should you download/upload a big database? It is too big for my laptop storage to download and then upload.
    - You could do this from the login node. 
    - If it is usefull for many, then you could request and we can set this up centrally.
    - Some examples of centrally setup databases are in `/cluster/shared/databases/`
        - ls: cannot access /cluster/shared/databases: No such file or directory 
            - The location is on SAGA, fram we do not have that
            - On Fram there are some shared databases here `/cluster/shared/`
    - I did not find it, i specifically need the DRAM db
        - Can you please give some more details. A link to what this means and whether this is something that would be useful for others


### Queue systems

6. I like the "Do Try This at Home" :-)
    - Thanks! :D

7. Backfilling is like "this table is reserved but they won't arrive before 19:00 so you can use it until then"
     - That was a very good way of putting it!  I'll "steal" that -- Bhm. :)
     - Yes. This is to use the overall system in an optimal way.
     - A job that is submitted later than a job that is ahead in the queue and that will not affect the future schedule jobs, should start now, isn't that fair?
        - yes.

8. Should we prefer mpirun or srun in job scripts for MPI-aware calculations?
    - This depends on the MPI flavour you use, see the link below for details
    - There are some documentation on this, because it depends on the cluster: [click here](https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html?highlight=mpi)

9. How do you know how many tasks you have/need? 
    - The first thing to consider is to see if your program can use more than one core(task) at the same time. A program that cannot do this will not go faster by pumping more cores
        - So different tasks will run on different cores?
            - Yes, Slurm (in our setup) will hand out one core (or more) to each task - and will never hand out the same core to more than one task or job.
    - If your program can use more than one core at the same time, most programs has an optimal number. i.e. the processing will become faster with more tasks, then at some point it will not go any fast and even become slower
    - Some time we might have to change the code slightly . For this we have advanced user support . [here is as example](https://www.sigma2.no/enabling-code-for-hpc-through-advanced-user-support)
    - also more about that later today and also tomorrow. we will also practice this
    - Just to be sure, the number of tasks is different from using array=1-n in the slurm script?
        - yes, --array submits an array of individual jobs, while --ntasks is how many tasks each job asks for.


10. What is a good way to estimate how much memory is needed before running a job? 
    - we will discuss and practice tomorrow
    - but until then: https://documentation.sigma2.no/jobs/choosing-memory-settings.html

11. Some Slurm terms 
   * Priority: All jobs starts with same priority, then more you wait priority will increase
   * Requeue : After nodes (a compute nodes) was allocated, if this had to be changed( may the compute node whent donw) another node had to be used , this value should used along side *Restarts* 
   * EligibleTime: The time Slurm decided that your job is ready to start, the start time might differe due to other factors (e.g. requeue)
   * Partition: We have different partitions (https://documentation.sigma2.no/jobs/choosing_job_types.html#job-types)
   * NodeList: The compute nodes used for analysis
- Yes, please <3
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Local slurm concept documentation](https://documentation.sigma2.no/jobs/submitting/queue_system_concepts.html)

12. Where can we find the presentations used today?
     - Linked from here : https://documentation.sigma2.no/training/events/2022-11-best-practices-on-NRIS-clusters.html#schedule  


### Using the HPC resources effectively

13. What is the difference between mpirun and srun? 
    - see also 8) but as far as I understand, I look at "srun" as being "mpirun" with stuff around it that integrates well with the rest of the job but as listed in 8) whether to use one or the other depends on machine


14. Can the MPI jobs not be distributed to the number of hyperthreads? (64 on fram for instance?)
    - By default: no.  But you *can* override it, if you really want.  For most programs, using both hyperthreads will not make the program run faster.

15. about redirecting logs to /dev/null:
    - I would only worry about redirecting if your application writes a lot of data from many processes in many small chunks in little time intervals. otherwise it can be pity to not have logs when a computation crashes and we have then no idea why since the logs are gone.



###  Exercise 12:00-12:15 (demo)

  - Scaling Exercise: Copy the directory `cp -r /cluster/projects/nn9984k/best_practices/Day1/scaling  .`
  - Follow the instruction in `README.txt`
  - Use training project Account and reservation while running your exercise


16. So, the documentations says to not use mpirun on saga.. Is this case specific? (https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html?highlight=mpi)
     - good question. answering as staff [name=RB], I think our documentation needs to become clearer in this recommendation. It is for me not clearly enough communicated whether I should use srun or mpirun.
         - "mpirun can get the number of tasks wrong and also lead to wrong task placement."
              - yes and I wrote that sentence :-) [RB] since it used to be like that on some clusters. what I am unsure about is whether this is still the case. Personally I use srun on Saga. few years ago when I wrote that, with mpirun on Saga the tasks sometimes did not end up on the allocated cores and the computation took longer since they were fighting over resources and some resources were idling.

17. Is there any way to predict how well a program will scale from how the code is written or something? 
    - looking at the code might give an idea where bottlenecks might start appearing as we increase the number of cores but it is almost always easer to "profile": run and analyze and then analyze again and then look at the code with the information of a profiling run. sometimes, surprising bottlenecks might appear. as we increase the system size, different parts of the code will have different scaling behaviours and unexpected code portions can take over and dominate run time.
    - depending on which parallelizing strategy and tool is used, one can estimate limitations. for instance: if this a shared-memory parallelization (e.g. OpenMP), it probably cannot go beyond one node so the limitation is the number of cores that can access the same memory. 
    - if no parallelization strategy is used, then the code won't scale beyond 1 core and there we need to parallelize at the Slurm level



## Feedback for the day

- One thing you thought was nice
      - The last demo was nice. 

- One thing we should change/improve tomorrow and next events

