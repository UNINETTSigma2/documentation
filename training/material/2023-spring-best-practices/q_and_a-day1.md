---
orphan: true
---

(training-2023-spring-best-practices-notes-day1)=
# Questions and answers from Best Practices on NRIS Clusters May 2023, Day 1 (May 9)

```{contents} Table of Contents
```

## Schedule Day 1 (9th of May 2023)

- 09:00-09:10 - Welcome, Practical information & icebreaker question 
- 09:10-09:45 - NRIS Infrastructure (Einar Næss Jensen)
- 09:50-10:30 - Job Types and Queue Systems on NRIS Clusters (Bjørn-Helge Mevik)
- 10:30-10:45 - Break
- 10:45-11:30 - Using the HPC resources effectively (Ole W. Saastad)
- 11:30-11:45 - Break
- 11:45-12:15 - Summary and exercise demo (Ole W. Saastad)
- 12:15-12:30 - Q&A 


## Icebreaker questions

- What would you like to get out of this workshop?
  - learn more how to use HPC system for running AI analyses

  - Confidence in using HPC facilities
  - Other clusters than Saga, such as Colossus on TSD
  - More about #SBATCH parameters and how to estimate memory, time, CPU etc in a script

- Are you able to access Saga? Add your "o" to your option
    - yes: oooooooo
    - no: o
    - maybe: had to reset password
    - do we need to reset password again if we participated in the previous course?
        - I am not sure, but try to log in and check that it works

- Are you a member of the nn9987k project on Saga when you type cost?
    - yes:oooo
    - no
        - if no, where should people write to?
        - Can not login to Saga?
        - Do not have access to project after login
     The link to change password is : https://www.metacenter.no/user/   

## Questions and comments

1. how long does it take before new password is working after reset?
   - 15 minutes or less (as far as I know)

2. When are the recordings uploaded?
    - usually after a week or two
      - we will send out an email when it is ready


### Questions to NRIS infrastructure
- Do you know what GPFS is
    - yes: o
    - no:oooo
    - https://www.ibm.com/docs/en/gpfs/4.1.0.4?topic=guide-introducing-general-parallel-file-system
- Do you know what S3 storage is
    - yes: 
    - no: ooooo
    - https://www.javatpoint.com/aws-s3
- Do you know what Infiniband is?
    - yes: o
    - no: oooo
    - https://en.wikipedia.org/wiki/InfiniBand


###  Questions/comments to Job Types and Queue Systems on NRIS Clusters 
 
3. Comment: restaurant equivalent to Slurm backfill: "the table is reserved from 21:00 on but you can have it until then for a short dinner" :+1: 
    
4.  When working with creating models I need to work more interactively for debugging and all that. How can that be done? I have been using Jupyter on the UiO system earlyer. Spyder would even be better. is this possible?     
    - great question. for these situations it is impractical to submit a job and wait few hours for it to start. for this you can submit an interactive job where you get an interactice allocation on a node.
    - more info here: https://documentation.sigma2.no/jobs/interactive_jobs.html
    - inside an interactive allocation one can then start an editor or run debug/test jobs until the interactive allocation stops

5. Can Jupyter be connected to this HPC-ressources?
    - here you mean: it would be running in the browser on your computer and connect to a jupyter server on the cluster? 
       - Yes, prototypes with small amount of test data
       - it is possible to run jupyter on the computer and connect it to a jupyter server via SSH. it is pity that we do not have good documentation on how to do that where I could point you to.
       - These pages might answer parts of this questions; https://apps.sigma2.no
    
6. What is mpirun? vs srun
    -  For most practical work choose the one you like, if you need the options mpirun offer you might use mpirun, but avoid the -np option, this is set by SLURM. The performance using mpirun or srun is the same. For large jobs with a huge number of mpi ranks srun might yield a faster startup. Both mpirun and srun can start any script, just multiple instances of it. When there is no SLURM e.g. interactive work you need to use mpirun, so if you want you can always use mpirun.
    - but when do you use it(either of them)? I probably missed the intro to it
    - more info here: https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html

7. How to estimate number of tasks/processes? Is it always just one task unless you do parallell work in the job? 
    - we will have a session on this good question tomorrow. summary of the session: it typically requires calibration and trying few configurations to know which is the optimal number for that type of job. once you know, then you can keep that setting for the series of jobs.
    
    
### Questions/comments to Using the HPC resources effectively

8.  Is random access also bad for solid state disks?
    - Random access is always bad, but how bad depends on the media. Random write is better than random read. (For quite obvious reasons?!)

9. Does $SCRATCH or $LOCALSCRATCH exist also on Colossus? Or what is the equivalent storage location there?
    - I do not see anything related to a $LOCALSCRATCH here: https://www.uio.no/english/services/it/research/sensitive-data/help/hpc/job-scripts.html#Work_Directory, but $SCRATCH seems it is on a fast system, so might be equivalent in performance as the $LOCALSCRATCH on the other systems
  
10. When is it useful and/or necessary to use MPI programs? Is it commonly used for a single person performing "normal" data analysis?
    - Any large scale program is MPI based, it can run with distributed memory. There is no limit of how many nodes you can run on. Take Betzy, at 1341 nodes and 172k cores one can (I have) run jobs using over 100k MPI ranks.    
    - it is one out of many mechanisms to make use of several cores. but it is not the only one. can you describe a bit more how your data analysis is? because whether it is useful/necessary, it "depends" what language/tool/library
    - I usually use several bash scripts to do serial calculations. Bioinformatics
      - and the bash scripts call python or R scripts/tools? 
      - yes, a mix of python and R, loading modules 
         - in this case, often mpirun is not so interesting but one parallelizes rather by splitting the problem into independent batches and one can parallelize using slurm jobarrays or similar. but some R and Python packages use OpenMP or MPI underneath and it can be difficult to know/see from the outside. Tomorrow we will have a session on this. Example for a Python package which can use OpenMP under the hood: https://documentation.sigma2.no/jobs/choosing-number-of-cores.html#code-may-call-a-library-which-is-shared-memory-parallelized
     - High volume production is also parallel work, but falls into the class of "naturally parallel" and hence gets little attention.


### Questions/comments to Summary and exercise demo 

11. Path where to find the demonstration project:
    - ```/cluster/projects/nn9987k/resources/scaling```

### Feedback

- Please name one thing that you liked about today
   - ...


- What should we change/improve for tomorrow or next time?
   - ...
   - ...
   - ...
