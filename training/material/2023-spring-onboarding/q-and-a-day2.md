---
orphan: true
---

training-2023-spring-onboarding-notes-day2)=

# Questions and answers from HPC on-boarding Apr 2023, Day 2 (Apr 20)

```{contents} Table of Contents
```
## Post Workshop Survey 

- Please give your feedback [here](https://skjemaker.app.uib.no/view.php?id=14765086)

## Icebreaker Question

### What challenges do you anticipate or have you faced when inheriting scripts/inputs/examples from other persons?

- example that the author provided was incomplete
- the inherited example was too large (took too long to run and queue before knowing that it even works)
- example referred to files/paths that I had no access to
- hard-coded paths, abundance of dependencies
- variables, function names and comments written in a human language I don't understand at all
- poor or no documentation, including what variables are supposed to be
- Missing dependencies, such as other scripts or packages/libraries which no longer exist/are valid
- no comments (documentation missing) to explain what the script or the particular parts does

### Do you know what conda is?

-   Yes: oo0oo00000
-   No: o000
-   I'm an expert:
-   Snake: oo~
-   Yes, but not in a unix system
-   Mamba is better i think

## Sessions
https://documentation.sigma2.no/training/events/2023-04-hpc-on-boarding.html

-----------------------

## Your questions

1. what does executable python mean? And are the path variables permanent?
   - if you type `python` and hit enter, you start the Python shell. but what really happened is that you ran an executable called python located at a certain path. using `which python` you can find out where it is located
   - some path variables are defined when you start the terminal (when you log in), and then you can change the variables (typically using `module load`) but the changes only live for the duration of your terminal session. once you close the terminal, it will go back to the defaults. the take-away from this is that it is a good idea to do module loads inside job scripts so that the job scripts (and thus path variables are correctly defined inside job scripts) become more reproducible

2. what if the python version that you need is not in the system?
   - First is to make sure this is the case, by using ```module avail python ``` 
   - Then if it is a newer version you could send a support request (more on this later today) asking the module to be installed
   - Alternatively, you could use the Miniconda module to create your own python environment
   - If the version is old or leads to errors that you can not resolve, when trying with Miniconda, you can use a container. We have Sigularity installed on our system (more on this during the best practices course)
       - https://documentation.sigma2.no/training/events/2023-05-best-practices-on-NRIS-clusters.html

3. do you need to remove the standard env as well?
   - typically no. we modify the environment by adding variables "to the left" and the system searches paths from left to right until it finds it so it does not matter if standard variables are somewhere "to the right" and never used.

4. So the number AFTER the package indicates the dependencies in some way or have I misunderstood?
   - in the modules we often see two number sets (example: `gnuplot/5.2.8-GCCore-8.3.0`). the first refers to the version of the package (here: 5.2.8). the second refers to the version of the so-called toolchain (here: 8.3.0). toolchain is the compiler that was used to build gnuplot from its source code. here both gnuplot and GCCore may have other dependencies on other modules.

5. Not really a question: `man module` results in the error `man: manpath list too long
   - That is a bit weird. From what operating system/tool are you logging in? Asking because I do not see this error on my computer. 
       - @login-2.FRAM ~]$
       - What os on your laptop?
           - macOS 12.0.1 (21A559) (does my OS affect the remote login node?)
         - problem is locale ... looking for a solution.   
           - please try this: `export LC_ALL=en_US.UTF-8` and `export LANG=en_US.UTF-8` followed by `man module`. if that helped, then you can add the two exports into your `.bashrc` on the cluster. 
           - yes; there are some settings that are inherited - specially from zsh on mac and using iTerm2/terminal. However, this problem does not exist with Visual Studio Code. 
        ```
        @login-2.FRAM ~]$ man module
        man: can't set the locale; make sure $LC_* and $LANG are correct
        man: manpath list too long

        ```
6. I was able to load an older version of Ruby, which caused GCCcore to be reloaded with an earlier version (7.3.0). However, all other modules are still dependent on GCCcore 10.3.0. Is there a way to prevent me from having loaded the "wrong" version of Ruby?
   - Try not to mix modules compiled with different compiler, this will lead to errors hard to debug. 
   - You need to find a module combination that are compiled with the same compiler/tool chain
   - If you give me the list of modeules you want to use together, I can show you how (if that is possible)
       - That's OK, I just wanted to fail the bonus exercise and see what would happen if I tried to load an incompatible module :)

7. `module avail node/` returns `No modules found!`: Can Node.js be made available?
   - Which means this module is not installed and not been requested. You could aske it be installed. Send request to support@nris.no.
   - When you send the request, please include a short description on how it will be used  
   - note that you can also use EasyBuild to install it for yourself. this is not part of this course but there is documentation on it. this can be a good solution to install software that only one person or group needs
   - You could also use Conda/Miniconda to install this (follow the ongoing lecture )
       - Conda to install Node.js? Interesting 👀. Will also look at EasyBuild.

   - If you load the latest R package you will also get node.js . The package is hidden, but it is there. If you load : module load R/4.1.2-foss-2021b
then you will see : which node
/cluster/software/nodejs/14.17.6-GCCcore-11.2.0/bin/node
        - Interesting, thanks!

8. Can you explain again about GCCcore versions?
    -  It basically means that there are some similarities between the free gnu ompiler suite and the commercial/costly intel compiler suite. Codes that have the GCCcore toolchain can be used together with both codes that has the GCC toolchain and the intel toolchain as long as the GCC version is the same.

9. What does the ml in ml purge do?
    - see `ml --help` - it can either list or load or unload. personal opinion: I find this inconsistent and surprising and I avoid it and prefer to type `module list` or `module load` explicitly.
    - thank you!

10. about the meaning of `mkdir -p`:
```
man mkdir :        -p, --parents
              no error if existing, make parent directories as needed
```
    - As always, man is your friend.
    
    (apart fron when log in from new mac and trying cost --man ;-))

11. It was told that conda can help others reproduce my work. How does this do this?
    - the reproducibility aspect of Conda is that not only I can install my dependencies into an isolated environment, it also makes it easier to communicate to others who need the same environment on their computers by sharing the `environment.yml` file which lists all dependencies and the version. personal recommendation: I always create Conda environments and always install packages from the environment.yml file. In other words, I first document the dependency in the environment.yml file and then install from that file. Then all I need to do is to share that file and others can install environment from it. This makes reproducibility for others and my future self a lot simpler.
        - Great! What needs to be specified to the reader in order for them to be able to reproduce it? the environment.yml file? does it contain all the packages(versions) that were used?
            - a reproducible environment.yml will list packages and versions (and also channels where to install them from) example for an environment.yml file: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually

12. is pip install not recommended here?
    - using pip install in combination with virtual environments works similarly and is also fine if your software dependencies are Python packages. it is the same idea. the instructor now discourages using pip but they mean pip in combination with conda. it is perfectly fine to use pip in combination with virtual environments (not mentioned in this course).

13. if I have already made a pyhton environment in the home folder, how to move to project folder?
    - you mean now conda environment of virtual environment?
        - Yes, in my own user not the course user (in SAGA). Update: Sorry conda environment in SAGA!
           - what I would do, assuming that you installed the environment from environment.yml (conda) or requirements.txt (virtual environment), then I would remove the environment and re-create it in the new place. I highly recommend to work as if the environment was disposable and could disappear at any moment. This will help you in future and make your computations more reproducible. If you are unsure which dependencies and versions are in the environment in question, you can export them into an environment.yml file.
               - So you have a fresh export of your conda environment in case it disappears that you can activate again?
                   - I always have an environment.yml and install from it. Then I always know my dependencies. But if you don't have it yet, you can use the export function to generate it from an "undocumented" existing environment. In other words I regard environment.yml as the documentation of my dependencies. It is very valuable to have.
                       - Very good idea! I would like to do that! Could you share some code on how to create this file?
                    - try these two and compare the results (they can be useful in different ways):
                      - `conda env export --from-history > environment.yml`
                      - `conda env export > environment.yml` 
                          - Thank you very much!
                             - pleasure! it has been after years of suffering and not remembering what I did that I started always installing from files :-)
                             - if you want to ask your colleagues an uncomfortable question in the corridor during coffee break: "which package versions does your notebook/project depend on?" if they have a hard time answering, then the project might be hard to re-run in 2 years.
                                 - Yes this is very good to know. So follow-up question. How do I activate the environment from a file? Normally I do 'conda activate my_env'. Do then do 'conda activate my_env_file.yml'?
                                    - you activate it the same way as before (from environment, not from file) but I only install from a file. So I personally never do "conda install somepackage" (unless I am sure I don't need to remember this) but always do `conda env create -f environment.yml`. so the file enters when creating and updating the environment, not when activating. then if me or somebody deletes the environment, I don't care and I just re-create it from the file.
                                        - Ah ok I see! Excellent, thanks so much!

14. Just missed the take home message: conda should be installed in our home environment and not in ".."?
    - Try to avoid the HOME area to store conda environments as the space is limited there 

15. Where should we continue excercising after this course? The fram access will be closed after today it seems.
    - The target audience of this course are the future users. So we expect that you will apply for a project or join an existing one after the course. 
    - I understand this means you are not part of another Fram project with some compute quota?
        - Not yet, but I have applied to Fox - guess I could continue there for now? But I would like to experiment with GPU access, which I understand Fox does not provide.
    - Fram does not have GPUs but SAGA and Betzy does. 

16. You might face the below error when trying to activate a conda environment on our systems  
    ``` CommandNotFoundError: Your shell has not been properly configured to use conda activate To initialize your shell, run... ```
    - Use the following command (after loading the anaconda module) and before activating the environment 
        - ```source $EBROOTANACONDA3/bin/activate```

17. When trying to access one of the [archives](https://md.sigma2.no/HPC-Onboarding-Spring-2023-day-1) I'm presented with a login screen: What credentials should I use? - I don't seem to be able to register.
      - oh that must be our fault. checking .... fixed.
          - 👍
      - indeed they are inaccessible. we are working on moving them somewhere else where they are visible and will update here and on top of this document once done. thanks a lot for reporting.
      - Can you please try now 
          - now it works
          - 🙏

18. What is the cluster/work/users/USERNAME  area for compared to cluster/home/USERNAME?
     - the /cluster/work one is good if I need to store many or big files for a short period of time (e.g. temporary file that are used in a calculation but not meaningful 2 weeks later). /cluster/home is typically for files I don't want to lose but the space here is limited. Downside of /cluster/work: not backed up and also it removes files older than 20-ish days.
        - Very good to know, thank you!
        - RB: personal opinion and not everybody will agree: /cluster/work is a good place to store Conda environments (but store the environment.yml files in /cluster/home)
        - OWS: /cluster/work/users/$USERNAME is not persistent, files are deleted after a pre set (subject to change) number of days. 

19. What are groups? Are they subsets of projects, or above that in the hieracrchy, or something else?
    - in unix/linux, groups existed before Slurm projects were invented. so technically they are independent but on our systems they roughly map to each other. try the command: `$ groups` and it will list all groups you are in and you will also see group(s) with the same name as the Slurm project.
    - command not found (Windows)
        - hmmm! interesting - maybe this is only something staff can do?
            - Found the issue, just writing 'groups' works

20. Can someone write the code that was used to change ownership. I missed it. 
    - `man chown` or  `man chgrp` gives you information about that command
        - e.g. ```chgrp GROUP_NAME MY_FILE```  (Replace GROUP_NAME and MY_FILE to match the group and the filename/filepath)
        - e.g. ```chgrp -R GROUP_NAME MY_FOLDER ``` (for recursive settings use -R)
            - Thank you!
    
21. Can you please explain how to estimate how much memory, time and CPUs are needed to run a job? Also how many CPU hours needed for a project
    - This a hard question to answer, the correct answer is normally found by trial and error. 
        - Ask SLURM for a lot of memory, run a short run (but full data set), review the SLURM log.
        - Run with increasing number of cores and record the scaling.
        - Explore interactive on an interactive node. 
   
22. This might be out of the scope of this course, but do you know if there's any alternative to the `dusage` command in fox? Since I found `dusage` is not available in fox. 
     - [author of dusage]: this is a tool we wrote for Saga/Betzy/Fram and it is indeed not a standard unix command but I do not have access to Fox so I am unsure but somebody who has will answer ...
         - Oh that's dope! Thanks for creating this, I found it really handy :)
               - thanks! suggestions on how to improve it most welcome
   - `man du` estimate file space usage
   - Fox documentation about disk usage
       - https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/work-directory.md

23. How to find out how much my project is charged? and how much a run costs - and what it depends mostly on?
    - number of cores times time used (not time requested, but the scheduler does not know how much you will really use before the job is over)
    - better than "number of cores used" is to think in terms of "number of cores reserved" if you ask for much more memory, you may block cores that donate the extra memory from being used by other users
       -  BUT HOW MUCH MONEY approximately? 
          - pricing info (but who actually pays, depends on the project): https://www.sigma2.no/user-contribution-model

24. I haven't fully understood the difference between CPU and GPU hours. Could someone please explain briefly? And in which cases I need to specify the different ones?
     - (we have reached out to staff colleagues for an answer ...)
     - 1 GPU hour = 16 CPU hour - price also depend on the category of your project.

25. If I start a job that requires more CPU hours than i have in my quota. Does the job run and then I get a bill for the amount i exceded or doesn't the job run?
     - the job won't even start (it will refuse to be submitted) - this can be important when requesting too much time for a job that is much shorter since slurm cannot know how long the job will actually take. it can also affect colleagues in the same project. so asking for excessive amount of time and cores can prevent my colleagues from submitting jobs until Slurm realizes that my job did not take 1 week but only 5 seconds.
        - Thank you!

26. Could you repeat the GPU vs CPU hours?
    - https://documentation.sigma2.no/jobs/projects_accounting.html
    - 1 GPU hour = 16 CPU hour - price also depend on the category of your project.

27. just a question regarding software installation, are you going to cover singularity ?
    - my understanding is that not part of this course. but please still do ask questions about it and we will try to answer.

28. So, the difference is computing power (CPU vs GPU)? You're saying charge, but I assume you get extra computing power as well? Or you are "getting" x16 hours allocated of CPU?
    - CPU (central processing unit) and GPU (graphical processing unit) works differently. In general, GPUs are much better at churning through large amount of computations that are heterogeneous as it can do this in parallel. If you have a workload that could benefit from this, you will get MUCH more than 16x performance. If not, it might even be slower. You should investigate if the program you are using could utilize a gpu or not.

29. What is the difference between `<user>` and `<user>_g`? 
     - they are defined as different unix groups but I admit I don't know why it was done this way (my colleagues who configured this are not part of this training event). I think it was done as workaround to allow disk quota.
       - So we dont have to do anything about it? I did not really understand the changing of ownership for allocation purposes
          - the information to keep is that on Saga and Fram moving files from one part of the storage to another is not reflected in `dusage` which confuses everybody who first sees this and the reason is that quota is owner based on these systems and not location based. and the fix is either to wait for our scripts to fix this for you overnight or to change ownership but luckily we don't need to understand for this the difference between user and user_g. takeaway: when moving from home to project folder (this is the most typical scenario), also change ownership to the project.

30. Can you briefly explain $SCRATCH and how/when to use it? (sorry, I have not been able to join for all sessions if you have mentioned it earlier)
    - `$SCRATCH`  is a variable that points to the work/temporary area. For some systems this is default set, for our systems not. In my calculations, I typically set it up pointing to what corresponds to `$USERWORK`; meaning `/cluster/work/users/$USER`.

    - OWS: On some systems $SCRATCH is set to point to a NVMe storage, this might be very beneficial for many jobs.
        - Only $LOCALSCRATCH is mounted on NVMe (on Fox $SCRATCH is on NVMe)
            - OK, on NRIS systems, the above is true =)
    - Always use $SCRATCH (or $LOCALSCRATCH) if running batch jobs, these variables are set up by the sysadmins to facilitate the best possible performance.
    - However, on Saga - you want to move files as little as possible. So if you are sure that your temp files are not violating the quota - do the whole calculations in projects area. 
    - I guess the short answer is no; we are not able to explain briefly...
    - Try to schedule file movement, if reading an input file once from project area there is no need to copy it to $SCRATCH, typically is read/write scratch files and anything with a lot of IO or metadata operations. 

31. In what order do you try increasing? Do you start increasing cores? or memory? or nodes? etc.
    - Sometimes they are not independent
    - Increasing cores are almost always better than nodes if there are more cores available on the node you have =)
    - First you need to undestand the program, if you have that knowledge then it is easier to avoid the bottle necks.
        - If you do not know about the program, try with two cores to see if it can run parallel before going any further with cores. Then if you hit the core limit of one node. Try with two nodes and make sure the progfram could handle distributed memory. i.e. you see that the program is using both nodes. If it can hanlde that then you could increase node count.
        - You could use the following command at the end of a job to see how much resource it used
            - seff JOB_ID
            - example
               ```seff 12345```
    
32. Are `nodes` and `cores` interchangeable terms? Or what is the difference?
    - A node is a separate computer. A computer has a CPU, and the cpu will have a number of cores. Each core is capable of doing calculations independently of the other cores, but it has very fast communication to the rest of the cpu.
    - For instance; each node on Fram has 32 cores. So a node is a "larger" unit than a core. Like your laptop; has typically 1 cpu/node but many cores. 

33. What is considered a 'large job'. At what point should we contact you discussing how to set it up so as to avoid blocking for other users
    - I was talking about transfer of really large data. A large job could be a job with many thousands of cores and there it could be good to contact us for advice. Some jobs are so large (so "wide" in terms of cores and nodes) that we need to schedule a reservation for them to make sure that they can start but for most application users do not need to contact us before selecting job parameters but they can if they are unsure about optimal use for them.
    - Basically, the slurm workload manager will do quite a good job at sharing the compute nodes. Each project will get a certain amount of cpu hours, so if you use them up, you are no longer able to run jobs. If you have a job that takes up a lot of nodes over a long time, your job will have to wait in queue for a long time. This way, if you are "greedy" you will get "punished" by longer queue times. If you have a problem you want to get solved, that you think could run more efficiently, you can contact us for help on doing that.
    - There are also limitations on how much resources a job can reserve. If you for some reason needs to reserve more than the maximum amount, you will need to contact us.

34. Can you say a bit more about how you use chatGPT to ask for errors? Just copy/paste and it will answer what it is? Or how do you use this tool for coding questions?
    - Said during the presentation - Just ask it /type your question (answered by a participant, not organizer).
    - Be careful because if it doesn't know a function for something it will make up one that sounds right and use it as if it were real. But asking followup questions or providing it with other suggestions can refine it to be more usable. E.g. ifit makes up a function X that doesn't exist, you can then ask it how to write a function that would do X (answered by a participant, not organizer).

35. What should we contact sigma2 about and what should we contact NRIS about?
    - NRIS: technical questions
    - Sigma2: anything about account
      - But not login problems on clusters

36. When I will do my first project using slurm on HPC, is there some way to ask some of you to "hold my hand" / lead me through the process the first time? (I don't have any "local colleagues" with this knowledge)

37. Should I try to use all CPUs on one node before trying more nodes?
    - It also depends, lets say you need more memory than one node has and the program can work on distributed memory then you should use multiple nodes
    - Basically "if you are not wasting resource" it is OK
    - Please note sometimes when using multiple nodes you might see slow down in performance 
 
38. Can someone please link some info to how to use 'devel' (or what was it called?)   
    - https://documentation.sigma2.no/jobs/job_types/fram_job_types.html
    - https://documentation.sigma2.no/jobs/job_types/saga_job_types.html
    - https://documentation.sigma2.no/jobs/job_types/betzy_job_types.html 

39. I got an e-mail from Sigma2 saying my access for project nn9987k expires today. Is it possible to know the time? Can I also get this kind of info from the command line on Fram?
    - Time will be midnight
    - I can log in to metacenter.no, but I immediately get shown the change password page, so I guess I can't see anything else.

41. Just out of curiosity. How much computing power do these clusters actually have?
    -    https://documentation.sigma2.no/hpc_machines/betzy.html has a teoretical peak of 6.2 Petaflop/s.
    -    https://documentation.sigma2.no/hpc_machines/fram.html has a teoretical peak of 1.1 Petaflop/s.
    -    https://documentation.sigma2.no/hpc_machines/saga.html has a teoretical peak of 795 Teraflop/s.

42. My supervisor (who has very limited programming knowledge) is the PI of the project (as it is in another field). He wishes me to have the "daily contact" with sigma2 and NRIS, is that possible?
    - Your host institution may offer a more focused hands-on tutorial. From which institution are you?
    - Daily contact with Sigma2 and NRIS personnel is somewhat problematic since we are not staffed with that in mind, but we will be able to help to a certain point. Note that it would from a pedagogical perspective also be much more useful for you to try on your own. We will monitor on general basis the system anyway and we have a well working support line. 

43. Is it possible to get a small quota allocated to do some testing of the system on my own (before applying for a project) to see if i will have the sufficient knowledge to actually use it?
     - Soon I hope tha answer to this is yes. We are working to get control over one of the compute projects for recruitement, tutoring, coursework and testing. At UiT there is an acccount for this, and ideally it should at least be one for each partner university in NRIS. What university are you from?

44. Where can the lecture recordings be found?

45. Are you going to cover how to use other clusters during the advanced course (like Colossus for example, working inside TSD)? 

### Slide deck: https://bit.ly/help-with-supercomputers

(please keep questions coming)

https://documentation.sigma2.no/getting_help/qa-sessions.html

#### Feedback
##### Somethings went well
  - I liked the markdown document (HedgeDoc). Me too x3!

  - Thank you for a fantastic course! I learned a lot, and got very thorough feedback on all questions I have. Everything was presented in a very basic and understandable way. The hackmd is very nice to ask questions in. And the fact that you are enough people to answer quickly and thoroughly is very very helpful. Then it is possible to keep track with what is currently being presented while asking questions.
  - I really liked the well written training pages, who was mostly easy to follow, written quite clearly.

- Gave me information and the confidence to get started.
- Very nice course. Well explained for a new-beginner. 

##### Somethings to improve 
  - It was sometimes difficult to follow what code was written in by the instructor (to try on the go myself). It would have been nice to have a more elaborate description to follow on e.g. the storage/usage part. 
  - The intro to conda and python was very nice. It would also be nice to get an intro to how to run R scripts on HPC. If it is mostly the same (except the conda environment), a short comment on it would be nice. 
  - Agree with first comment, sometimes the coding was a bit fast. Some of it is of course written in the documentation for the course. But not all, and then it would be nice to have some time to write it down before executing the code and the command getting lost in the output. 

- Could be useful to go through a "hands on example" from start to finish. Maybe at the very end of the course after the "basics" are explained. Could be useful to just see one example.
- I think the vscode session should be part of the normal session. This was a very useful tool. 
