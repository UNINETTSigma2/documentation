---
orphan:true
---

(training-2023-spring-best-practices-notes-day2)=
# Questions and answers from Best Pratcices on NRIS Clusters May 2023, Day 2 (May 10)

```{contents} Table of Contents
```

## Schedule Day 2 (10th of May 2023)

- 09:00-09:10 - Welcome, Practical information &icebreaker question 
- 09:10-09:50 - HPC Best Practices & Trouble shooting (Ole W. Saastad)
- 09:50-10:00 - Break
- 10:00-10:50 - How to choose the right amount of memory + Exercise (Radovan Bast & Jørn)
- 10:50-11:00 - Break
- 11:00-11:50 - How to choose the number of cores + Exercise (Radovan Bast & Jørn)
- 11:50-12:00 - Break
- 12:00-12:30 - Q&A & Discussion 


## Icebreaker questions


- Have you ever used any of these?(you can vote by adding an “o” at the end)
    - MPI: 00
    - OpenMP: o
    - POSIX threads:
    - Others, please specify below
        - Python multiprocessing: oo
        - Slurm jobarrays: o


## Questions and Comments HPC Best Practices & Trouble shooting

1. Slurm array jobs: do they all start at the same time once they start?
    * Not guaranteed to start at the same time
    * You should not rely on the start order of each one
    * As the jobs are identical and submitted more or less at the same time, they will start close by. This depends a lot on the cluster load and the resource request (how easy to fit the jobs in, considering current load)

2. Could you write down the most useful commands for troubleshooting here? 
   - great question. it would be good to have a brief summary here or somewhere linked from here.
       - In application/software does not start: Check if missing libraries using ldd.
       - If slurm not accepting the submission: `sbatch jobscript.sh` gives errors and `scontrol show job <jobid>`. 
       - If job does not start quickly: `squeue --me` gives info about all your jobs (and also `scontrol show job` as above).
       - If slurm is accepting the submission, but the job dies prematurely: Read the slurm-output file that is placed in the folder from where you submitted your job(s). Often the software of choice often has some log printings with feedback also. Read this carefully. 

3. Btw, where can we find these slides?
   - we are working on making them available.
   - Some of the slides are also available from previous courses, see https://documentation.sigma2.no/training/events.html#past-events (for instance the one from Nov 22.)



## Questions and comments How to choose the right amount of memory 
https://documentation.sigma2.no/jobs/choosing-memory-settings.html


4. Is it possible to know how much memory actually used ? you did ask for 3000M and the RES was 3G, how much actually used ?
   - the tool that will show you the "high water mark" (maximum memory ever allocated) is `/usr/bin/time -v`. The Slurm-based tools (seff, jobstats, sacct, Slurm output) use a 30 second sampling. Also unfortunately the Slurm-based tools have a bug on Saga after upgrade and they seem to be factor 2 too large at the moment. We are investigating why.
   - Example: https://documentation.sigma2.no/jobs/choosing-memory-settings.html#by-prepending-your-binary-with-usr-bin-time-v

5. Did Jørn mean `cpus-per-task` ?

### Exercise
:::info
__Hands-on until 10:45__
- Find the exercises on Saga at `/cluster/projects/nn9987k/best-practices/choosing-mem`
- Have a look at the `README.md`
- Don't forget to copy the files to your own home folder
- Solutions are in `SPOILER.md`
- Ask questions in the breakout zoom rooms or here in the collaborative documents
:::

:::info
__Survey__
Did you try the exercises? (put an 'o' down)
Yes, everything worked: ooo
Yes, but I had problems:
No:
:::

6. How do you escape from tail -f <>.out? 
    - CTRL + C
    I tried that but it did not work, neither did q or ESC. But it's fine, I just quit the terminal
        - are you on a mac?
            - PC

7. We found that Maximum resident set size (kbytes) is 3210916 kB and the memory that we use (from memory statistics, in GiB) is 6.1. The 6.1 said how much we will use or the maximum size is what we should consider in run.sh? Can you repeat what they said before?
     - 3.2 is the correct result here. there is some bug in Slurm on Saga in the diagnostics which shows too high numbers. so unfortunately 6.1 is wrong result but we will fix that.

8. Does "/usr/bin/time -v" work if you put it somewhere else in a bash script? (Not directly before an executable)
     - yes, it will measure whatever comes after it on the same line. you can measure different things in the same script.


How is the speed of the last session:
- too fast:
- too slow: o
- just right:ooo


## How to choose number of cores

This session's material: https://documentation.sigma2.no/jobs/choosing-number-of-cores.html
Overview over [slurm/SBATCH parameters](https://documentation.sigma2.no/jobs/job_scripts/slurm_parameter.html)

9. If I want to access more memory, should I ask for more cores, even if I dont use them.
    - Ole actually touched on this yesterday. He suggested asking for more memory and not more cores. 
        - Isn't there a limit one core can access ?
    - Yes, and if it goes beyond the highmem nodes, then that I guess would be your only option, so all depends
    - you can tune with --mem-per-cpu= as long as you are within the resources on the nodes you ask for.

10. Is not this testing also using CPUs? So you have to pay for this testing multiple cores (scaling test). Or is there a way to test your code without actually using up compute nodes?
    - Your usage of this testing is very small and should not take any significant amount of your quota. If this is a problem, ask for a larger quota. 
        - Have either of you tried to ask ChatGPT about this? To estimate without testing
            - if someone has written about the specific code you might be lucky.
            - Often this is something that is either problem specific or not known. Since ChatGPT are not currently build for future-telling, we have to do it the old fashion way...



### Exercise
:::info
__Hands-on until 11:45__
- Find the exercises on Saga at `/cluster/projects/nn9987k/best-practices/choosing-cores`
- Have a look at the `README.md`
- Don't forget to copy the files to your own home folder
- Solutions are in `SPOILER.md`
- Ask questions in the breakout zoom rooms or here in the collaborative documents
:::

11. How to run all scripts simultaneuosly? 
    - to submit all in one go: `for file in *.sh; do sbatch $file; done` (using bash's for-loop to iterate over all files that end with .sh)

:::info
__Survey__
Did you try the exercises? (put an 'o' down)
Yes, everything worked:oo
Yes, but I had problems:
Still working/waiting:0
No:
:::

## General Q&A
12. ...

13. ...


## Feedback

- Name one thing that was good?
    - Everything was perfect! Exercises are important for us to practice :) Thanks!
    - ...

- What should we improve for next time?
    - ...
    - ...