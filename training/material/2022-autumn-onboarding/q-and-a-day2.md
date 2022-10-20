---
orphan: true
---

(training-2022-autumn-onboarding-notes-day2)=
# Questions and answers from HPC on-boarding Oct 2022, Day 2 (Oct 20)

```{contents} Table of Contents
```

## Icebreaker Question

- Which software would you like to run on our HPC machines?
    - Custom-made MATLAB and Python scripts
        - +1
        - +1
    - codes for density functional theory calculations (Elk, Quantum Espresso)
    - Fortran 90, including OpenMP (for parallel computing)
    - Julia
        - +1
    - more Rust (but it already works and works well)
    - The "Parallel Ice Sheet Model" (using PETSc, MPI, C/C++)
    - Nextflow, Docker, SNAKEMAKE
    - COMSOL Multiphysics
    - Seismic unix
    - Madagascar
    - R scripts
    - Cogenda Genius
    - Oracle database / SQL
    - LaTEX
    - Matlab parallel computing


- No MPI ? Most of the above is single node applications not really Supercomputing stuff.
   - [RB] It looks like the trend goes towards workflows that run many smaller jobs, also one-node applications. It can still be supercomputing though. Or maybe rephrasing: I think we need to provide place both for calculations that require lots of super fast network and calculations that don't. And this is also why there is not only one cluster but we try to provide 2-3 clusters at the same time.
       - Could you expand on that? E.g. which is the best cluster for MPI? Or point to relevant docs. Thanks!
           - Very rough answer: extremely many cores: Betzy. Many cores: Fram, very few cores: Saga. But as below, it depends and please reach out to support if you want to apply for an allocation and are unsure to which cluster.
           - [Ole] depend on job size (# CPUs). MPI is ok with all clusters.
   - [Ole] High volume production is in many cases quite hard to accomodate, huge I/O load. We have 3 systems, with inreasing job sizes Saga (single thread to some tens of CPUs), Fram medium size, Betzy large, up to 172000 CPUs, and for special requirements LUMI (CPUs+GPUs).


## Questions


### Accessing software

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html

1. Question from yesterday: is there any limit for files transferring from/to NRIS'HPC (file type, file size)?
    - There are [indeed some limitations, on size](https://documentation.sigma2.no/files_storage/clusters.html#overview)
    - we will talk about those limitations after the break today
    - but in short, there is no limitation from the network point of view (that I would know of, other than network maybe breaking if a transfer takes days) but there are limitations on how many files and their size sum you can have on the clusters (more later)

2. Where could we report bugs in the module system? For instance a missing library: `module add netCDF/4.9.0-gompi-2022a; ncgen`.
    - email to support@nris.no; this will then be forwarded to the software install team at NRIS who are in charge of helping users with software installations. the actual problem might be somewhere else but then they will know where to submit an issue if that is not a problem we at NRIS created.
        - Thanks, I reported the issue.
    - For now the solution is to also load `bzip2/1.0.8-GCCcore-11.3.0`, but this is indeed a problem with the module!
        - And will use this workaround!

3. Is there an overview of the keywords available? Or are keywords just any string which might be a substring of a module?
    - The "keyword" command searchs any string that is in the module description, it is slightly missleading in name
        - Thanks, I see.

4. Should the course use examples that would work on all the clusters? blast was not available on fram
    - It is often easier to use a single cluster, just because of such differences, so it is recommended that you follow along on Saga
    - hmm yes maybe we should change the example. tricky but not impossible.
    - The clusters can in some cases have slightly different versions available, but here you will learn how to check what modules are available with module avail etc. And if something is missing you ask support support@nris.no :) 
        - Yes, I agree. I found out that I had to use another module or log into saga. Just thought that the examples could use something that is available on all the clusters
        - Yes, best in principle, but suddenly for some reason one module in the examples might be gone or not there in another cluster, so yes, in principle the best way to try to ensure that the example modules can be found on all clusters. 

5. So, what is a toolchain / why is it needed? 
    - a set of compilers that were used to build the software. a compiler translates ("compiles") readable source code into runnable executables and libraries. when we load a module that depends on other modules and other libraries, we often have to make sure they were all built with the same compiler set (toolchain)
    - "why is it needed" is a really good question. it was a choice by the designers of EasyBuild (the framework we use to install most of our software) and I guess their motivation was better reproducibility? unsure but there are other opinions and approaches on how software dependencies can be managed
        - Most software distribution methods use a shared version compiler to ensure ABI (application binary interface) and layout compatability of the software being compiled

6. [MP] Question that came up here in my break-out room from yesterdays session: SLURM gives information about how much memory was used maximally - but how can a user find out how much memory was used in total for the job - i.e. to know what to ask for next time? 
     - but isn't "max" the thing to ask for since it will never go beyond that? or max plus 15-20% perhaps is a good approach. adding 15-20% because it might not be always the same and it would be pity to get the job cancelled after 2 days of running just because few MB were missing. But it is a good idea to not ask for excessively too much "just to be on the safe side" and we will maybe comment on that later today
        - yes, obv, did not think - it is of course! :) 
    - Try the command `seff <JOBID>` after the job ends [link to documentation](https://documentation.sigma2.no/jobs/choosing-memory-settings.html#using-seff)

7. If you need non-standard python packages, where should you load them?
    - We recommend that you use virtual environments and install packages there. Our packages contain the most used libraries, optimized for our system (like `numpy`), but we can't install all users desires.
    - [Tutorial on installing Python packages as a user](https://documentation.sigma2.no/software/userinstallsw/python.html)

:::warning
Comment now in voice: do not put any `module load something` into your `.bashrc` because it will make your job scripts less reproducible for you in future and also for support staff helping with problems. Much better is to place all `module load` into your job scripts. You could also place a `module purge` in the beginning of the job script to make sure you start with a clean environment.
:::

8. SAFE (https://www.uib.no/safe). Is SAFE a more secure wrapper on top of the HPC? In other words, is there an HPC backend for the SAFE system? -or- SAFE is just a remote system with more focus to security?
      
      - There is not a HPC backend connected to SAFE, if you are working with sensitive data and need HPC then TSD is the cluster that is suitable. At UiB we have this flow chart to show the difference between compute solutions:  https://www.uib.no/en/it/155472/it-research

### Exercise until 10:05

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html#using-software-modules-in-scripts

(the green box there)
- If you have time you can submit a job and use the training project `#SBATCH --account=nn9984k`

How is it going in the rooms?
- Room 1:o
- Room 2:oo
- Room 3: :white_check_mark: 
- Room 4: :white_check_mark: 
- Room 5: :white_check_mark: 


### Storage and compute quota

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/compute-storage-quota.html

We will go through these two:
- https://documentation.sigma2.no/files_storage/quota.html
- https://documentation.sigma2.no/jobs/projects_accounting.html


9. Any way to use Visual studio code on a folder on the server?
    - I don't use VSC, but maybe [the SSH extension](https://code.visualstudio.com/docs/remote/ssh) could work?

10. Does the compute and storage quota work the same on Saga and IDUN?
    - I don't know about Idun, but at least it is the same on Saga, Fram, Betzy and LUMI
    - Saga, Fram and Betzy are the national resources, while Idun is a local cluster to NTNU

11. What are the main criteria for allocating computing time? How to write a good application?
    - It depends on the size of quota you apply for. Huge applications like 10++ Millions of CPU hours require very good track record. Small applications 10000 CPU hours etc require a simple application. Applications in the 10k-100k scales do not need a track record. 
        - Thanks! What would be a good track record? E.g. demonstrate scalability on sigma2 computers? Scientific output? Code used on other supercomputers in the past?
            - Yes, publication record from CRISTIN (https://www.cristin.no), scaling demo, in all these cases (10++ Mcore hours) we know the PI. If you're new it is a bit more work to get such a large application approved. Start small and work youself up. 
                - I am on a short-term position (3 years), not new to supercomputing but new in Norway, does that mean that I should consider applying outside Norway?
                    - If you work at a Norwegian University etc you a free to apply. How many core hours do you want ? 
                - I think 100k-1M core-hours per year, in one or two years time.
                    - 50-500k core-hours per semester (period) should be easy. Just go so sigma2 web (https://www.sigma2.no/high-performance-computing) page and start the process (you can apply at any time, but expect delay and only very small quota outside the deadlines). 
                - Thanks! The webpage states that applications are evaluated based on scientific excellence. Is the scientific part more important than doing benchmarks, performance analysis? What should be the ratio between scientific justification and justification for the compute budget in the proposal?
                    - Your application is so small that it will be ok, you have some track record (publications + HPC experience), then it should be no problem. If you are at UiO or NTNU you could use local resources.

12. So what would be a solution?
    - If you have too many small files, e.g. in your output data - you could try to rewrite the way you store the output data - maybe it is possible for you to make larger fewer files. Sometimes it is not, but other times you can. 
    - there is a senario where the data provider is transferring data on a daily basis with large number of files everyday and we need to analyze the data on the go, even if the files are zipped we need to unzip and work with them, what would be the best practice here? 
        - At least unzip them in e.g. the project folder or work folder where you will have a larger quota than on your home folder for example. If there is a very good reason why you need a larger quota, you need to contact nris@support.no and request that - and then we will see if a bigger quota can be given or if there are other ways to solve it
    - Ok, what is the quota on the project folder? i cannot see that with dusage.
        - can you not? I see for example: 
        ```/cluster/projects/nn9984k       yes      52.0 KiB   1.0 TiB         13   1 000 000```
        where 1.0 TiB is the space quota and 1 000 000 is the number of inode quota
     -got  it, thanks!     :white_check_mark: 


14. Folders also have inodes, right?
    - yes also a folder will take up one "card"

15. is pri and nonpri used when asigning quotas? Is that something you specify when applying?
    - You get nonpri if you have applied for quotas after the deadline.
    
16. when and where the videos for the course will be available?
    - next week, you will get an email with the link when we upload the videos

### Using shared resources responsibly

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/18-responsibility.html

17. So, how do we make sure we are not running large jobs on the log in nodes? When we send a job using sbatch, does that automatically directed to computing nodes or should we do something particular?
    - If you use `sbatch`, `srun` or `salloc` then the computation will always run on the compute nodes
    - If instead you execute your program manually - not using the slurm commands, but just calling the executable, you will be running on the node you are at, which in this example is the login node. 

18. For jobs like zipping and upzipping large number of files, would you recommand doing it on the log in node or computation nodes?
    - For a large amount we recommend that you do it on the compute nodes, either through `srun` or often [easier `salloc`](https://documentation.sigma2.no/jobs/interactive_jobs.html)
    - For small jobs that finish in a few seconds the login node can be appropriate

19. Regarding proxy, is it possible to build snakemake envs for a workflow in the jobscribts. My experience is that it is not possible to download trohough conda dynamically
     - There is probably a way to do that which I don't know. You can write a support ticket to us with the query support@nris.no.

20. How does one directly run commands on a login node?
    - Just by calling the executable like we showed earlier e.g. `./my-executable.sh` so typically manually - not using the slurm `sbatch` for instance 

21. A follow-up to question 20, how one does the same thing directly on a computation node?
    - Use [an interactive job with `salloc`](https://documentation.sigma2.no/jobs/interactive_jobs.html)

22. The tar is taking a long time, maybe because you are running it on login node? Would be nice if you show how to run the same job on a computation node.
    - Yes, but still much quicker to do tar and scp and then untar compared to scp all the files separately!
    - If you are generating large number of files and then creating an archive. I could reccomend to use the local scratch. Parallel file systems (like /cluster) could be slow when handling very large number of files.
        - on a login node you could access this as follows
        `[SAGA ]$ cd /localscratch/ `
        - [On a job see here](https://documentation.sigma2.no/jobs/scratchfiles.html)



### How/where to ask for help

[Slides](https://cicero.xyz/v3/remark/0.14.0/github.com/bast/help-with-supercomputers/main/talk.md/)


### Feedback for the day

- One thing you thought was good:
     - Thank you for a very useful course. I have learned a lot. 
     - I learned a lot. Thank you! I'm looking forward to the more advanced course. 


- One thing we should change/improve next time

    - Feedback form should use radio buttons, not checkmarks...













---
Please always write above this line, at the end of the document
