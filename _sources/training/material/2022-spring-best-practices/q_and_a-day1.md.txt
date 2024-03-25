---
orphan: true
---

(training-2022-spring-best-practices-notes-day1)=
# Questions and answers from Best Practices on NRIS Clusters May 2022, Day 1 (May 23)

```{contents} Table of Contents
```


## Icebreaker question

- What would you like to get out of this workshop?
	- (Add answers below with a leading -)
	- How to install custom python repositories / environments on the cluster. +1
	- How to write code that takes advantage of this cluster (parrellel programming, GPU)
	- How to calculate CPU and memory for my job
	- How to best manage my data (200gb video files) +1
	- How to best manage my data reading (20 TB, 2 million files with 2-100 MB, and I'm reading ~300 files ~= 2 GB at a time)? What are the best practices for reading such data, what are the don'ts?
	- How to make my code faster ;)
	- How to compile open source software on a cluster
	- How to do data analysis and view the plots in FRAM?




## Best Practices and trouble shooting

1. Now that we learned that we should probably not use hyperthreading, how do I turn hyperthreading on or off?
    - Hyperthreading is turned off b in Slurmy default. On the norwegian HPC systems if you ask for cores (e.g. ntasks), you get allocated physical cores.
    - If you know what you are doing then you may enable this with "--ntasks-per-core=2" SBATCH parameter
    - [More Details](https://documentation.sigma2.no/jobs/job_scripts/slurm_parameter.html)
    
2. Can you define what exactly you mean with random read?
   - the opposite of a random read would be to read "at the end" or read in the same order as the data was written
   - random read would be to read "somewhere", not necessarily in order
   - in other words during a random read, the reading device needs to move from one place to another and cannot keep reading and thus spends a lot of time "moving" and not reading

3. "Try to not copy files": but somewhere we also hear/read to not run from HOME. So should we rather keep them on HOME and run from there?
    - The comment about copying is referring to parallel file systems. Which means your HOME or project area are the same in that sense (on the /cluster partition).
    - If you have millions of read/write, then using the localscratch (see the question 4 below) would make it more efficient. 
    - This you have to plan, i.e. there is no universal answer, e.g you have pne big file and you are planing to access it millions of time during then better to copy it to local scratch (Q4 below). Copying that files inside the /cluster to somewhare else would not make a difference.

4. What is a local scratch? 
    - Local scratch is a local storage on the compute node, hard drive, Solid State disk, NVMe etc. 
    - More abot this: <https://documentation.sigma2.no/jobs/scratchfiles.html>
    - all other storage is accessed via network but "local scratch" is a hard drive that is inside the same "box" as the CPU and memory and has thus much faster access
   
5. What does MPI mean?
    - message passing interface. it is one of several techniques/protocols to run a code in parallel. it is suitable for problems where the independent "tasks" need to communicate and where they are not completely independent.
    - if the jobs/tasks are completely independent, then MPI is typically not a suitable tool to parallelize but job arrays might be more suitable
    - Not offitial explanation, but how I try to think of it: On a compute cluster compute nodes are connected using fast network, so data can be moved fast between them. However memory/RAM of each compute node are isolated. i.e. if you assign a value to a variable in memory on one compute node (e.g. A=10), other compute nodes would not know it. So MPI can help you make this connection, i.e. let your program know and use what is going on on a nother computer on the cluster. Then MPI helps to consolidate the results at the end as well. 
    - All the executables launched by mpirun are identical, they get asigned a rank and total rank number. The only way they can communicate is by sending a message fram one rank to another.
    - As programmers generally need to know to program in parallel the MPI programs tend to scale better than OpenMP/Shared memory. 
## Exercise scaling 

- Copy the directory `/cluster/projects/nn9989k/nris-best-practices/`
- `cp -r /cluster/projects/nn9989k/nris-best-practices/ .`

- inside there, consider only the `Day1/scaling` part
- The current exercise is in the directory NPB `cd Day1/scaling/NPB`
- there is also a README.txt with instructions
- the other two exercises below `Day1` are for later today

6. Where do we run the exercise today ? 
    - meaning on which cluster? on Saga
    - but if there is any problem there, the same exercise folder can also be found on Fram
       - I logged into the folder in saga
       - once on Saga you can copy the exercise folder to your home: `cp -r /cluster/projects/nn9989k/nris-best-practices/ .`
       - Yes its done

7. Whats going on? I dont hear anything
     - currently exercise session until 10:15 (see above for instructions)
     - you can join one of the breakout rooms (bottom right in Zoom window)
     - we will resume after the exercise

8. Should we be using Fram or Saga? I did the scaling exercise on Fram but maybe it's better to use Saga instead?
    - our primary cluster for this course is Saga but Fram is "backup" if there is some problem on Saga but same account and reservation is put in place on both
    - but it's no problem that you ran these on Fram

9. What "Mop/s/process" stands for? I got the following: 

   - slurm-4nodes.out: Mop/s/process   =                 2035.09
   - slurm-9nodes.out: Mop/s/process   =                  1836.85
   - slurm-16nodes.out: Mop/s/process  =                   1178.52
what's the conclusion from the result?? More nodes then less operations?
       
   - The purpose of the exercise was to measure the performance for different number of tasks of the simulation. Its measure is Mop/s/process, where Mop/s = million of operations per second. 
   - The results above show that the performance decreases with increasing number of tasks, which is bad, meaning that to use resources optimally you should use fewer tasks. This is a very small simulation, and it does not really matter if it takes 20 or 50 seconds. For realistic simulations (think hours or days), you should do some tests to find out a good balance between performance (Mop/s/process) and the time needed to run the simulation.

10. How do I make plots in linux? (related to the scaling task)
    - many people use Python/matplotlib or R/ggplot2
    - there is also gnuplot

11. Can you share the plotting technic used
    - I will ask the instructor after the session to share the script, I think it is a gnuplot script
    - The simple solution is using a spreadsheet

12. Command to explain later
    - `ldd <binary>`, `strings <binary>` (read man pages : man ldd and man strings)


## Choosing memory and core settings

We will follow:
- <https://documentation.sigma2.no/jobs/choosing-memory-settings.html>
- <https://documentation.sigma2.no/jobs/choosing-number-of-cores.html>

13. What is the command to check the quota required for a submitted job?
   - compute quota: `cost`
   - storage quota: `dusage`
   - There is no quota "for the submited job". The quota is on how many CPU hours you have remaining in the project and how much storage space you have left.

14. How can I specify that I want 16 tasks on specified nodes?
    - You shall also specify the nodes : e.g. `--nodelist=c1-2,c1-3` (assuming you have the right to run on these nodes)
    - wasn't that done in the example? would that be ntask-per-node? Yes, nodes=1 and ntask-per-node=16 since Slurm multiply nodes with ntasks-per-node it will allocate 16 cores on the node.
    
15. I have tried to follow the info on localscratch, but I am unable to get a value in ${LOCALSCRATCH}. On Saga. Like
```
#SBATCH --gres=localscratch:2G
echo "localscratch: ${LOCALSCRATCH}"
``` 
gives me `sbatch: error: contain a NULL character '\0'.`
 - Have you got an example script that runs this?
 - Try : /cluster/projects/nn9989k/nris-best-practices/Day1/slurm/localscratch.run  
 - Thank you. 
    
16. The command "seff" doesn't give any information on GPU usage I see, or is there something similar for GPU usage (other than the log files themselves)?
     - I think seff don't compute GPU effficiency according to the `seff` code [seff](https://github.com/SchedMD/slurm/blob/master/contribs/seff/seff)
    - `sacct' combined with  'Reqgres`command can use to list the requested GPU nodes, may be not indices.
    - [sacct](https://slurm.schedmd.com/sacct.html) 
    
    - You can monitor GPU utilization, Please see our documentation [here](https://documentation.sigma2.no/getting_started/tutorials/gpu/cuda.html#step-3-monitor-the-gpu-utilization)
     
    
17. Could you explain CPU Utilized and CPU Efficiency and Job Wall-clock time on the seff output? How does one use this info to better structure job scripts?
    - If you use too many cores and waste time in admin the util goes down. Hence monitor util and check that it's not too low.
    - Job Wall-clock time is the time used by the job. CPU Utilized is the time used by all CPUs, i.e. each CPU ran x Job Wall-clock time.
    
18. Is it possible to say anything general about in what type of computation it can be expected that concentrating the cores to a single node is beneficial or not? 
    - Concentrating the cores in a singe node is always preferable if possible. Slurm will do this by default if resources are available. 
    - RB: I don't fully agree. I normally prefer to not concentrate them until I know that it has any effect. Letting the scheduler distribute them gives it more flexibility and less queue time for me. It can matter but in my experience more often than not it does not matter for smaller calculations.
    
19. Which module and version did you load for the `perf-report` tool?
    - <https://documentation.sigma2.no/jobs/choosing-number-of-cores.html#example-for-a-memory-bound-job>
    - it was `module load Arm-Forge/21.1`
    - then prepend the command with `perf-report`


## Exercise (15 min)

- Copy files from here: `/cluster/projects/nn9989k/nris-best-practices/`
- Consider only the `how-much-memory` and `how-many-cores`
- how-much-memory:
  - first run the script as is: `sbatch run.sh`
  - then try to find out how much memory binary-a is using
  - use <https://documentation.sigma2.no/jobs/choosing-memory-settings.html>
- how-many-cores:
  - first run the script as is: `sbatch run.sh`
  - then try to find out optimal number of cores/tasks for binary-b
  - <https://documentation.sigma2.no/jobs/choosing-number-of-cores.html>
- You can also try to analyze your own code or run script  

---
    
20. If you run a combined MPI & OpenMP job on Saga, how do you deal with he fact that there is a different number of cores on the different nodes?
    - hmmm. very good question. in this case you could ask for tasks per node for the node type with the higher core count? or how do others do that?
        - What happens then to MPI ranks that have less cores? Stupid question: Can they easily discover the number or cores avvilable to them?
        - Slurm allocate cores on different nodes in such a way that you get what you asked for. You may only get part of the nodes with more cores than on nodes with less cores. You should not have to worry too much about this. Slurm generally do a good job.
     
21. Could you please summarize all of the tools for finding optimized parameters? like seff...
    - good question. I will summarize in voice
      - memory: probably easiest is run job and then seff
      - cores: run a series of jobs and compare timings but make sure the job is not too quick as the comparison is then not conclusive. later drill more into the calculation with seff and perf-report 

22. Are different nodes with different amount of memory/core accounted differently at NRIS?
    - here is detailed information: <https://documentation.sigma2.no/jobs/projects_accounting.html>
    - in short: ask for what the job needs and you probably don't need to worry too much (except sharing the project allocation with other project members responsibly). the accounting can become a problem if you request much more resources than are actually used. also memory, not only cores.

23. Regarding reading data: You mentioned a few simple do's and dont's, but is there any more advice / tools for taking care of tricky file reading / databases?
    - E.g.: I have a database of 20 TB on Saga, 2 million files with 2-100 MB where I'm reading ~300 files ~= 2 GB at a time (16 parallel read processes seems to help here to speed up efficiently). But should I check that the 300 consecutive files are well distributed across disks rather than just the same single disk / RAID so that I'm not limiting reading because only one disk / RAID is accessed at the same time?
        - I expect ypu are on Saga. yes
    - considering I'm running an array job on e.g., 4-8 nodes, using all CPUs on each node - anything else to consider?  - The DB is just files in a specific structure. Each file is is read into memory as a whole (so no random read I think) so I don't think Localscratch helps. The question is more on how to avoid bottlenecks in case the data is all on the same physical disk / raids, or what to consider there.
        - If you do a lot of work on each file, like random read you could move the files to localscratch. 
        - Why do you need to copy the files, are you extracting files from the db ?
        - Point is that you might overload the db, does it scale ? There is a limit on how many requests is optimal for the db.
        - The more you read from the db the more of it will be cached in the FS and new reads might be in the cache. With such big db the files would be spread over a number of RAID sets, hence you should get good parallel performance. Try to run scaling test when reading from the db. Mayby you are optimal, of maybe even more threads could perform better. - Ok thanks!

24. Is there a maximum number of cpus you can ask for in a job (on Saga)? 
       - For Saga job limits please see the [documentation](https://documentation.sigma2.no/jobs/choosing_job_types.html#saga)

25. Is using the cluster mostly a good solution if I need more cores/gpu's? If my job needs just a few powerful cores and 200gb RAM is it better to just use a personal workstation and buy the RAM?
    - cluster may be good if you need many cores. the key is really the "many", not the "fast". often the CPUs on a cluster a slower than on a laptop but there are many more.
    - - Is the cluster good for jobs that need a high amount of memory but a low number of cores? What is the maximum amount of RAM per core?

26. Where can I find the next Q&A session (the monthly event)? 
   - [NRIS Monthly Q&A](https://documentation.sigma2.no/getting_help/qa-sessions.html)

27. Is there a tool like top/htop that can tell me about how busy the file system is? I have experienced that my Python scripts take > 60 s to import packages (usually less than 2 s), so I can only guess that the file system was slow / busy?
    - we are aware that sometimes it can be very difficult for users to see whether it is slow because of something they do or slow because of the file system (slowed down due to to other reasons). we are working on setting up better dashboards with health checks so that it becomes easier to see.
    - currently the best option is to check <https://opslog.sigma2.no/>

28. Are the results of the exercise correct?
    - yes. great job!


### Results from exercises

- what did you find out? what was the memory requirement for `binary-a`? 
    Memory Utilized: 2.24 GB
    Memory Efficiency: 65.44% of 3.42 GB
- and what `--num-tasks` would you recommend for `binary-b`?
        Nodes: 1
        Cores per node: 4


### Feedback Day 1

One thing you thought was good:
  - The second session was very practical and interactive --> that was very helpful +1
  - Enjoyed the content of both sessions. The instructors were good at pointing out what to think of when tailoring the slurm scripts for a particular simulation. `strings` was a discovery for me too :). Nice you have several tools documented to check CPU and memory usage. I like the `time` wrapper a lot. Used `perf-report` before, a favourite too.
  - Not so much to do with today, but the docs on <https://documentation.sigma2.no/> are really getting very good!
  - The demo using the terminal was nicely made. Appropriate tempo. The user/directory coloring and extra command window at the bottom were helpful.

One thing that should be improved until tomorrow or next time:
  - The first part was very abstract, and then very specific. A more middle approach with more simple practical information with more context would be much more useful +1
  - Some of the terminology was already to advanced for me (e.g. MPI, localscratch...) +1
  - The first exercise would need more explanation for first-time users. There is a typo in `scontrol show job $jobid` (it was with capital S). A bit less scrolling maybe when presenting the content on the website.
