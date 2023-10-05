---
orphan: true
---

(training-2023-autumn-onboarding-notes-day2)=
# Questions and answers from HPC on-boarding Oct 2023, Day 2 (Oct 5)

```{contents} Table of Contents
```

## Icebreaker 

- Which software would you like to run on our HPC machines?
  - LAMMPS 
  - anvio workpackages, phyloflash
  - a python project
  - a python program that interacts with a JS client
  - trimAl
  - GTDB-Tk
  - R studio
  - Obitools 4



## Accessing software

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html 

(questions about previous days also welcome)

1. What if we need more than one software ? e.g Python and R
    - in this case we can load both as a module but we need to make sure they are compatible. more about it in a moment.

2. How do I know which software are not campatible?
    - This is very good question and something we must be very considerate. Because if the modules that are loaded together are not compatible this will cause issues
    - The main consideration is the compiler or the tool chain used. This we can see from the prefix(last part of the module name) of the module.
    - Another thing is when we load modules, it will load all the dependencies along side as well, e.g. sometimes we do not have to do anything extra
    - We will show more latter in the lesson

3. Still talking about more than one software, if both need to be in a container, can I download one container for one and install the other software inside the same container or should I use two different containers?
    - I would install both into the same container. Composing containers is not so easy but it is possible.

4. What is the difference between installing a software and loading a module?

    - Loading a module means that the software is installed as a module on the cluster, you can load them when you need the software, this is applicable to most cases, see [here](https://documentation.sigma2.no/software/modulescheme.html). In somecases, the software that you need may not be available as a module, then you can install it as a user. More on this is [here](https://documentation.sigma2.no/software/userinstallsw.html)
 
 
 5. Why did you remove parts of the path?  Did not understand that step and what it did/why it was done?
     - in practice we will not do this manually but we will let the `module` commands modify path. they will in effect pre-pend and modify PATH and thus make software visible or not visible. key point here is that linux will search through PATH from left to right.
 
6. How do I find more info about the module command, man module is not giving anything 
        - Try `module --help`
        - also: `module show SOMEMODULE`
 
7. The ```StdEnv``` is a set of settings that are needed for the module system to work. Loading this will help the system what to do where to find the software when you ask quesitons like ``` module  avail ``` from the system.

8. Do you normally make environments in saga to use software, like miniconda?
    - in this case I would module load the miniconda module and then use miniconda to manage dependencies
 
9. Can we use different toolchain for a job? 
    - As far as I understand the module system (or EasyBuild which we use underneath) assumes that we have a consistent toolchain in our environment
    - To avoid conflict I would suggest to go for same toolchain

10. what is GCCcore ?
    - it is one of the "toolchains". in other words it is a set of compilers that were used to build the software and dependencies. in this case it is the GNU C compiler. not sure what "core" is but it is probably a minimal version of the compiler suite.
    - `core` concept is from easybuild to serve as base for different toolchain.
      - thanks for clarification! 
    - You can find more info about toolchain on easybuild documentation [here](https://docs.easybuild.io/common-toolchains/)

11. Is it better to use Conda for installing software or a Container? 
    - both have advantages and disadvantages. I would start with Conda and only later consider a container. containerizing might be more work. but the advantage will be that it will be easier to move from one computer to another. but conda alone is probably simpler and it is good enough for most applications.
    - advanced answer: personally I use both using https://github.com/bast/singularity-conda (but I wrote it)
    - using a container can be a workaround around the problem of too many files or too long installation process
    - documentation on how to use Conda on HPC: https://documentation.sigma2.no/software/userinstallsw/conda.html

12. Is there a way to search for specific information about the software you have installed? 
    -`module avail` will list all the software installed on the cluster
    -but information about what the software does? l
    - `module show software`, for eg : `module show R/4.2.1-foss-2022a` will print alot of info about R and the version on screen - and also info about what the module do in terms of loading paths etc.
    - try `module help`, for instance `module help Python/3.11.3-GCCcore-12.3.0`

13. How can a JS client interact with a Python program running on Saga? More concretely, I assume that a web API first authenticates incoming requests from the client and then takes some actions on the cluster, such as running the Python scripts or submitting a Slurm job. But I would greatly appreciate it if you could tell me a few things regarding such a specific way of real-time interaction with the HPC cluster, particularly about the authentication. I think this is an essential use case concerning the "Accessing the Software" part of the tutorial. Thank you!
    - This is not the way a HPC cluster is normally used. HPC systems are best when you can breakdown your analysis to smaller components and anaysise the components in the same time (parallel)
    - From your question it seems that you have an idea what you want as a function, but your technical requirement may have to be discussed further. For example JS clinets run on the browser and they do not directly communicte with back-end (e.g with server)
    - Regarding authentication. Setting up your own web application to authenticate against SAGA is not recommended. What could be an alternative is to use continuous integration(CI/CD) pipleines. But for this even you need to define your front end better.
    - Can you please give an example on what your intentions are . 
       -  Thank you for the response. A good example is a user study scenario where the user interact with the browser app. The app sends the input arrays to the Python program on the cluster, where (1) the data is stored and labelled, and (2) a trained model generates a novel output based on the user’s input, which is fetched back by the client program and shown to the user. 
           -  For this I do not think SAGA is the best place to host this. But we have other resources like the service platform.Please send us a mail and we can have discussion about this to how we can help you. May be we can have some pipline, e.g a Git ci/Cd where jobs are triggered on a HPC system. 
           -  As shown yesterday, the interaction with the cluster is not immediate, e.g you ask the schedular (SLURM) to exceute the jobs when resources are availble.So a user interface directly communicating might have to wait for some time, that can not be predefined (e.g. it depends on current load at the moment the job was sumbmitted).  
               -  Yes, that was precisely what I have been concerning about. Thank you for the explanation! I will surely contact you and we can discuss more specifically so I will not spam here more:) Again, thanks a lot!




## Exercise

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html#id1
### Status:
- I am done: ooooo0oo
- I need more time:
- Not trying: 


## Compute and storage quota
- https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/compute-storage-quota.html


14. What does it mean to change the group using the command: chgrp?
    - on linux/unix system each file or folder has two ownership aspects. it belongs to a user and a group. it is possible to modify the ownership with chown or chgrp. did this clarify? (maybe I am answering out of context; I was in a different meeting)
    - Thank youf for the answer. The context is `dusage` and moving to the work directory, I am used to chmod and understand what it does, but not chgrp. Maybe you cloud provide a bit more clarification.
        - with `chown` you can change ownership. `chgrp` you can change ownership of the group part (ownership always has a user part and a group part)
            - What if you do not have the permission to do that? I meant chmod not chown? (Difference between chmod and chgrp)?
                - You can change permission for files you own or part of the group (and the group permissions allow this)
        - One example for groups is projects. Where a set of group members can share a folder

15. What are the different groups? How do we know which one to change to?
    - try typing `groups` - this will list you all the groups that you are part of
    - but what is fex the difference between username and username_g?
        - this is a very good question and I think the answer is that this is a technical choice made by the administrators to enable storage quota in the network file system. I don't know otherwise why we needed two different groups: `user` and `user_g`. It confuses me as well and I write this as staff.

16. Is the directory /cluster/work/users/myuser empty automatically every once in a while? I think they mentioned something related to that. So I should move my stuff to the project directory if I want to keep the data?
    - yes, files and folders older than typically 21 days will get automatically deleted: "The $USERWORK autocleanup period is at least 21 days and up to 42 days if sufficient storage is available". so this is a good place to run calculations and to generate temporary data that you don't need to keep but all data that you want to keep for later you should move away to the project folder
        - Thanks! I will go right away through all of my stuff given I have never transferred anything yet. I have been using it for one month so I might be at risk =)
    - I will mention this in the next session.
         - we will try to communicate this better in future to avoid surprises. on the positive side: it can be a nice place also for experimentation where you don't need to clean up because the system eventually will.
             - Good point! 

17. So related to question 16, now my work directory is 364 GB big. Is it safe to transfer all to the project directory in one shot (say, with rsync)? Then I will clean up from the project directory
      - first step: check with `dusage` that you have enough quota for it in the project directory
      - then you can use rsync to transfer everything in one shot. even if the transfer gets interrupted, you can resume.
      - if you are sure that your connection does not get interrupted, you could also `mv` the folders but here I would perhaps do it in chunks and not in one shot. `rsync` is safer w.r.t. connection loss.
          - Understood! Thanks!

18. can you please show the project are to explain a use of groups
     - What is the use of group ownership. So we can share files with others 
     - A group ownership is where you can share folders and files, for eg`nn9970k` project can have files with `nn9970k` ownership under the project area. 
    - you can use groups to assign access to files 
    - each user may belong to one primary group (`username_g`) and the project group here `nn9970k`, you can try `groups` and find the groups you are in
    - Does it explains ?



## Poll: Try `dusage`
- Yes: oooooooo
- No:
## Poll: Try `cost`
- Yes:oooo
- No: 


## Using shared resources responsibly
- https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/18-responsibility.html
- Links Espen mentioned:
[Choosing correct amount of memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html#choosing-memory-setting)
[Choosing correct number of cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html#choosing-number-of-cores)

[Tool for creating job scripts](https://open.pages.sigma2.no/job-script-generator/)


19. Thank you Espen for a great analogy of a hotel. How do you explain interactive jobs in that case ?
     - what was the hotel analogy? (I wasn't here and missed it)
         - the relationship between a supercomputer and a laptop is like the relationship between a hotel and a house.


21. Referring to last lesson what does Quota (nonpri) mean?
     - You get nonpri if you have applied for quotas after the deadline 

22. What are the numbers you us with chmod? Before we used fex +x 
    - You can find a details [here](https://en.wikipedia.org/wiki/Chmod)

23. So we should manually change permissions to our own files when we place them on the shared project directory in order to "protect them" from the other project members? I mean that no one could accidentally delete or modify our files.
    - normally you don't have to worry about that. you only need to worry about ownership and permissions when moving files from e.g. home to project folder since that does not modify ownership.
    - but good point that if you want to protect files against accidental delete of other project members, you might need to adjust permissions.
        - Thanks!  So the files I create in my working directory are created with permssions -rw-rw-r-- . Given that I only have access to my working directory could I somehow set it to create the files with writing permissions only for me so that when I transfer them they keep that status?

24. How do I know what my job require to run? 
    - start with a small example, grow the example, and monitor how much time it takes and how much memory it consumes and adjust towards what you think is optimal
       -  thank you
    - Nice resources to read [here](https://documentation.sigma2.no/jobs/choosing-memory-settings.html) and [choosing number of cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)

24. what does it mean if you submit a job successfully but cant find it in the squeue -u $USER?
     - It means the job is no longer running (finished/cancelled/termonated) or not in the queue anymore.
     - You can find info about the finished job by using `sacct -j jobid` or `scontrol show job jobid` 
     - how do you find the `jobid`?
     - Type `sacct -u username` this will give you a list of recent jobs(finished).

25. Sorry, the question might be out of scope, but is there an equivalent command for seff in TSD?
     - I am sorry, I don't know.  `sacct -j jobid` and `scontrol show job jobid` are general slurm commands that will give lot of information. `seff` is an independent perl script.
         - Please try seff. The one person who developed seff has also been working on TSD, so it might work.
             - Not well formulated request from my side I see... I tried seff in TSD before writing the question. I got the message ‘command not found'
             - Ah, sorry - then it is not there. 
             - Thanks! Good idea. I will do.
             
26. What is the best way to share log files ? send as attachment 
          - You may send as attachment
          - Send us the full path (where they are) and infom us that it is OK for us to look at them 


### Feedback of the day

- One good thing about today
    - very nice support for the Q&A, thank you for a nice course!
    - It is one of the few workshops/courses where the time is correctly managed. I wanted to highlight this given all the people involved in the organization. You all had a great managment of the time. Not easy to find in this environment!
    - The shared document seemed to me a nice way to avoid oral interruptions which of course helps on the time management. It will also be super useful to access in the future together with the recordings of the sessions.
    - Thank you! It was a very useful workshop for me! You cannot imagine how much!!! 

- One thing to improve
   - would be nice to see an example on how you use software in the batch files



:::info
Please always write new questions at the end of the document, above the line above. Don't include any personal information, even name.  Be polite and use kind language. You are welcome to help out other users, if you could. This document is public !
:::

