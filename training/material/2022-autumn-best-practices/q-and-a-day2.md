---
orphan: true
---

(training-2022-autumn-best-practices-notes-day2)=
# Questions and answers from Best Practices on NRIS Clusters Nov 2022, Day 2 (Nov 2)

```{contents} Table of Contents
```


## Icebreaker Question 

- Have you ever used any othese?(you can vote by adding an “o” at the end)

     - MPI : ooo
     - OpenMP: oo
     - Others, please specify below
         - Python multiprocessing: o
         - Slurm jobarrays: oooooo
 
## Questions and Comments

### [HPC Best Practices & Trouble shooting](https://drive.google.com/drive/folders/1XkQeEzYv_aNWNSlVeZuRurHv5L8HgTIH)


1. What if the program I want to run is a script(not a compiled program), e.g. Python. Commands like ldd will not work in that cas.
     - One trick would be to see if the program/script calls any libaries that make parallel porcessing possible.
     - if this is a self contained script that does not call libraries, you could `grep -r` for text like `mpi` or `multiprocessing`
     - good point above but it can be difficult to inspect dependencies/libraries of the script as they may not be really "visible". what I would do and this is not elegant, is to launch it and I would ask `top`/`htop` and check how many cores are actually busy. Alternatively I would use `seff` after the job is done to check the statistics.

2. Yesterday we saw the `top` command, is it possible to see the full command used ?, e.g. if an argument like parallel is used. when i use top I see only the executables name
     - with top/htop you can see the PID (process ID) of each process. then you could ask `ps` to give you info about that process. example for a process 12345: `ps 12345`
     -  You could try `top -c` to show the full command. 
        - nice! I did not know that. that's much nicer than the "ps"-detour

3.  Where can you find the log/feedback from a job in IDUN?
    - We will get back soon on this (waiting for the NTNU colleague)  
    - Is IDUN using Slurm? (I am not from NTNU and I don't know that system)
        - apparently it does: see https://www.hpc.ntnu.no
    - Yes, IDUN use Slurm. Does that mean it works the same on IDUN?

4.  Can you please explain what *NUMA* means in simple terms 
    - non-uniform memory access. in simple terms: the memory is not equally far away (in terms of access time, access latency) for all cores on one node so if the job is very "memory-heavy" (many and frequent memory reads and writes), then the performance may depend on the core-placement of the job. so in this case it can make sense to verify on which cores precisely were the threads/tasks/ranks placed by the scheduler.

5. How do I check the disk access/IO of a running process.
    - good question. hopefully the speaker can comment/clarify. [RB] I don't really know how to apart from maybe running a profiler like [perf-report](https://documentation.sigma2.no/jobs/performance.html)
    - One command you could use is `strace`
        - `strace -p <PID>`
        - You could only monitor your own processes
    - Another command is lsof
        - `lsof -p <PID>`

6. I'm a bit confused abouts threads, tasks, and ranks. What is the difference between them? 
    - shared memory parallelization like OpenMP use threads. threads run in parallel, do something, and communicate via shared memory
    - then there is message-passing interface (MPI) where we rather speak of tasks or ranks (same thing, different names) where the tasks communicate typically not via shared memory, but by sending messages to each other and each of the tasks manages own memory
    - hybrid parallelization can use both tasks and threads. technically they are different but conceptually I picture them as very similar.

7. Could the slurm output file be exempt from the disk quota?
    - one could redirect slurm output to paths where you have no or more quota (check also `dusage` for available places)
      - but everything placed on USERWORK will get auto-removed after a certain period of time (21 to 42 days depending on usage)




### [How to choose the right amount of memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)

Plan for this session:
- 15-20 min discussion/demo/motivation/overview
- 15 min exercise which you can find in `/cluster/projects/nn9984k/best_practices/`
  - on Saga, you can copy the exercise to your home: `$ cp -r /cluster/projects/nn9984k/best_practices/Day2/choosing-mem .`
- 10-15 min Q&A and discussion 
- Use Training project and reservation (but this is already in the run scripts).
  - `#SBATCH --account=nn9984k`
  - `#SBATCH --reservation=NRIS_best_practices`


8. Question to all:
    - I run code that I wrote or co-wrote: oooooooooooooo
    - I run code written by others: oooooooooo
    - I just `module load` a module written/installed by others: ooooo

9. Could we just specify total memory (and not per CPU), so several CPUs can just handle memory themselves?
    - #SBATCH --mem=2G # total memory per node (Just tried, it seems to work)
    - I would still prefer to specify per CPU because then the tasks don't have to be on the same node (it matters to some codes but does not matter to most) and you give the scheduler more flexibility to place jobs on other nodes where there is room for them

10. What is AveRSS?
    - AveRSS = Average resident set size of all tasks in job (as per [slurm documentation](https://slurm.schedmd.com))

11. When you use seff on a job, there is a line called memeory efficiency. It gives us some percent of something that is slightly less than the requested memory (e.g., 3.12 GB when 3.2 GB was requested). Why is that? 
    - hmmm. could this be GB vs GiB? https://en.wikipedia.org/wiki/Gigabyte
        - Ahhh, yes, I think so. Or maybe not, 3.2 GB should be 2.98 GiB (?). 
           - but I am not 100% sure, I did not double check this

### Exercise until 10:50

on Saga, you can copy the exercise to your home: `$ cp -r /cluster/projects/nn9984k/best_practices/Day2/choosing-mem .`

- in there check `README.md` with `cat README.md`
- let us know here how it is going
- the same folder exists also on Fram if for some reason Saga does not work for you

- I get the following error when using Arm Performance Reports: Lmod has detected the following error: The following module(s) are unknown: "Arm-Forge/21.1"
    - resolved: Arm-Forge/22.0.3
      - oh. so maybe documentation is outdated? 
      - Try the command : module avail Arm-Forge , it will list available versions.

12. I see that there is a difference between MaxRSS in the "/usr/bin/time -v" and the one you get in the slurm output, why is that? 3210868 vs 3210488
    - I admit that I don't know why this is. Normally it could be the 30s sampling but not in this case since the allocation does not change (I know it in this case since I wrote the code). The good news is that we don't need to know exactly. I would always add 15%-20% more anyway so we don't need to know all digits but it is a good question.
    - so the "time" one reports slightly more? Yes
      - interesting. I will investigate later where the difference comes from just for my own education but can't answer it now before doing more digging and counting

13. I do not see breakout rooms? How do I join?
    - Thank you

14. When checking how your program scales with the number of tasks, is there any point in checking the same number of cores several times? Or should the variation be very small? 
    - we will do that one after 11 but it is fine to start of course.
    - in this case probably not and the result will be very similar but sometimes it can make sense to run 2-3 times and take some average if the runs vary slightly. when profiling benchmarks we often run a step many times and average but here we want to get a feel and find out an approximately good core setting. even after 1 run it will probably be evident that some core choices are completely unreasonable.

15. How do you go to the other (e.g., medium memory) nodes? 
    - e.g. https://documentation.sigma2.no/jobs/job_scripts/saga_job_scripts.html#bigmem
    - see also https://documentation.sigma2.no/jobs/choosing_job_types.html
    - Medium (392GB) and big mem (3TB) are both part of the `bigmem` partition. Depending on how much memory you are asking for your job will end up on the appropriate computer.

16. What is the memory efficiency shown by seff? Can it be improved?
    - maximum_used_memory / requested_memory
    - Memory efficiency can improved by lowering the amount of memory you are requesting. Sometimes you might need more cores, so it is better to check job efficiency in `seff` as well before you play around with these numbers.
    - In general you want to aim for a high efficiency but also leave a few persent (10-20%) in case the process suddenly uses a bit more.
    - also "seff" is not something we wrote but something we installed :-)


### [How to choose the number of cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)

Plan for this session:
  - 15-20 min discussion/demo/motivation/overview
  - 15 min exercise which you can find in `/cluster/projects/nn9984k/best_practices/`
    - on Saga or Fram, you can copy the exercise to your home: `$ cp -r /cluster/projects/nn9984k/best_practices/Day2/choosing-cores .`
  - 10-15 min Q&A and discussion 
  - Use Training project and reservation (but this is already in the run scripts).
    - `#SBATCH --account=nn9984k`
    - `#SBATCH --reservation=NRIS_best_practices`
    - note that for the 128 cores example the reservation does not work since it's larger than the reservation. for this reason I have taken it out of the exercise script.

17. How can I do these calculations for jobs that take weeks?
    - You cannot scale it down ? Less timesteps ? Less iterations ?
    - Yes, 1-2 timesteps or iterations and hopefully the timing can be extrapolated from that. or a smaller system size if the timing can be extrapolated from that. but even if all of that is not possible, I would run the first 5-10 calculations on varying core counts to get a feel before running the other 90 calculations. there will be some "waste" during calibration but globally it might save CPU hours hopefully? it is always a balance.

18. Probably a very basic question, but can you run code on multiple cores (several tasks) even if your code don't include any explicit parallelization? For example, writing code that only uses numpy?
    - Something like arrayrun ? N copies with different input ? Typical for Monte-Carlo simulations.
    - Arrayrun works well. But then it does not make sense to try to increase the number of cores or the number of tasks?
    - A threaded numpy library might help, but does your code spend much time in numpy ? 
    - Something like numba? I'm not sure exactly how much time is spent inside numpy operations, but I would guess a fair bit 
    - Print out a timestamp before and after call to numpy and you'll know. 
    - Try setting OMP_NUM_THREADS=4 or something and run a test with numpy, some of the numpy libs are threaded. Intel have versions that are multithreaded. 
    - I will try that, thank you. And that means that it is sometimes possible to run stuff on multiple cores even in pure numpy? But if you want more control parallelization libaries are good? 
    - Remember Amdahl's law, if you spend only a fraction of time in numpy it does not help much even if numpy was infinitely fast. 
    - An expert on this will update on how do it later today. He's not in the office now, but will attend to it later today. Watch this spot.
    - Thank you for the detailed answer. I think I'm missing a more general understanding of how code can run on several cores, so I was originally wondering if you could do that without using any special libaries. 



### Exercise until 11:55

- on Saga or Fram, you can copy the exercise to your home: `$ cp -r /cluster/projects/nn9984k/best_practices/Day2/choosing-cores .`
- Use Training project and reservation (but this is already in the run scripts).
    - `#SBATCH --account=nn9984k`
    - `#SBATCH --reservation=NRIS_best_practices`
    - note that for the 128 cores example the reservation does not work since it's larger than the reservation. for this reason I have taken it out of the exercise script.
- please let us know how it is going and if you need more time



19. Are these videos of the course giong to be available somewhere after the course? Because i heard Radovan said this is not being recorded and not be published, or did i hear incorrectly?
     - The videos will be available later. What Radovan mentioned is that if you are talking, we will edit the part before publishing.
     - They will be published in the docs under [training material](https://documentation.sigma2.no/training/material.html#training-video-archives)

20. I know that R is a single thread executor, does this mean that it can only use one cpu core per session? If this is the case running R code on HPC without parallelization does not improve anything, does it? For those using python isn't this the case with python? does your code run on multiple cores if they are available? 
    - yes Python and R are similar in this. can R use libraries that are parallelized themselves? I believe it can but haven't used R myself yet.
        - There are R packages that can start several R processes and send tasks to them (for instance the "parallel" package that comes with R - but there are many more).  Some use MPI, some not.  There are also R-packages that use multithreaded libraries.  In many installations, the basic BLAS library that R is linked to, is threaded (not sure about the NRIS installation here, though).
    - but even if it runs only on 1 core it can still make sense to run on HPC clusters if you can run many independent jobs at the same time instead of running them one after the other
        - This can for example be done with Slurm job arrays like described [here](https://documentation.sigma2.no/jobs/guides/job_array_howto.html) or [here](https://documentation.sigma2.no/jobs/job_scripts/array_jobs.html) 

21. Building on question 18… If I have a script that someone else has given me and I want to run it, how do I find out whether it can be run on multiple cores? i.e. if it is even possible to run it on multiple cores, not if it will improve the time. 
    - Ask the one who wrote it
    - seff
    - or send a question to support@nris.no and we help finding out
    - RTFM (Read The Fine Manual)
    - Thank you!


## Feedback for today
- One thing you thought was nice
    - This last bit on different ways of parallelizing was nice
    - Exercise are always nice
    - This document and the way you answer question is great. 
    - Thanks for a great course!
    - Thanks!


- One thing we should change/improve tomorrow and next events
    - More examples with actual codes and how they run would be nice here, something like a benchmarking code. 
        - Time is very limited, this is case by case. Please feel free to ask for help.

    - I would appreciate some comments on which parts of what we are learning are applicable to IDUN.
        - Idun is a local cluster at NTNU, they operate as they see best for them. They do not need to follow the operating scheme as the national systems does, there is one «Driftsorganisasjon» who run Saga, Fram and Betzy (At UiO we try to do it similar as the national).
