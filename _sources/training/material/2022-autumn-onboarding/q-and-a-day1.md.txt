---
orphan: true
---

(training-2022-autumn-onboarding-notes-day1)=
# Questions and answers from HPC on-boarding Oct 2022, Day 1 (Oct 19)

```{contents} Table of Contents
```

## Ice Breaker Question

- Have you experienced any computational analysis/tasks that was too big or too long for laptop or the desktop?
    - yes when memory on laptop wasn't enough
        - same
    - doing density functional theory calculations
    - doing transcriptomic analysis
    - computing glacier ice flow (Navier-Stokes) on large grids for long time scales.
    - yes, same here.It happens during RNA-seq analysis
    - large seismic datasets
    - LiDAR / SAR data
    - Petrophyscial data
    - yes, I need multiple cores to run my codes which I do not have on my laptop
        - same here


## Questions

### Why use a cluster

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/11-hpc-intro.html

1. Not sure where to put the day's questions, but is there a computational reason servers don't have GUIs?
   - this is the place for questions :-)
   - traditionally the reason was that it's easier/faster/cheaper to send characters/letters over the network than graphics but these days compute clusters are moving also towards graphical user interfaces such as Jupyter notebooks and similar computational notebooks, or running a visualization postprocessing graphically on the server

2. What will be a typical configurations of such a server if more than 40 people are simultaneously working on a project.
   - for many people on the same computer (login nodes): we need many cores and lots of memory. then on the compute nodes it's typically fewer users on a node.
   - if you are interested in our actual hardware specs, this is what we have on login nodes: e.g. for Saga: https://documentation.sigma2.no/hpc_machines/saga.html
   - And if you are interested in applying for resources then take a look here https://www.sigma2.no/high-performance-computing
   - And also for the cloud service https://www.sigma2.no/nird-service-platform
   - sorry but what do you mean with login nodes?
      - It is a server (computer) that you ssh to from a laptop/desktop.
      - will be clarified in a little bit but this is the "pizza box"/computer
        where we connect to, login to. when you connect to e.g. saga, you land
        on a "login node". it's a computer without its own screen. actually we
        have maybe 5 of those there? then from a login node you launch
        jobs/scripts which then run on compute nodes. instructors will say more
        about this soon.

3. I've heard about VNC, but what is it exactly? And how does it relate to working in a cluster?
    - this is a graphical remote desktop so it provides a desktop that looks like own computer but is really running on the server/cluster where you can open a terminal. I see it as alternative to putty/mobaXterm/terminal

4. Do you have an iLO for managing your servers, or what are your procedures to monitor your server as a system manager
    - what is iLO?
      - I am not sure but a http local link to monitor the server, i guess.
        - aha. we have various monitoring tools in place and are developing those: for instance monitoring file system load, network, the memory and CPU resource use. the clusters are a big complex system and on such scale things will fail on a weekly basis just based on statistics so monitoring is absolutely needed.

5. Is it possible to run graphics applications on a cluster (for instance, programs for viewing atomic/molecular structures)?
    - yes it is possible. if you are on linux/mac, connect with X11 support. if you are on windows, I recommend https://mobaxterm.mobatek.net/
    - however, running big heavy graphical applications on the login node can affect other users on the same node. these systems have often not been designed for this. if the data is big and difficult to move, it may be your only choice. if the data is tiny and easy to move, I would probably copy the data back to my computer and look at it on my computer. so "it depends" as in everything in computing :-)
    - We will try to address this when we introduce the scheduler, why the HPC is optimised for "crunching" and not much for "viewing"
    - It is possible to use 'service remote desktop' to run graphical applications see here for more details https://documentation.sigma2.no/getting_started/remote-desktop.html

6. In disk file command (df) one of the output is xfs itxfs same as nfs?
   - Good observation. This is the local disk of the login node. There are local disks on compute nodes as well. These are only visible to the node you are in, so if you place a file here other computers can not see it. But these are not using parallel file system, so a good place to stage data. i.e. place your large input files that will be accessed many times during an job execution, only from one node are kept.
      - but where can I see this? are we talking here about Saga or which cluster?
      - i dont see it in my terminal but on the training pages the output of df -Th  | grep -v tmpfs , shows type
  - try `df -hT /localscratch`
      - yes it shows me type

7. sinfo returns an untidy collection of numbers that is not easy to read through, i see the list of nodes but at the end there are long lists of numbers that is not informative, is that how it is supposed to look or there is something wrong here?
    - It is more or less correct, but we can recommend to use `sinfo --summarize` to see the information a bit more condensed
    - `sinfo` also has support for specifying specific parts of the system, e.g. if you want to see the status of nodes in the `accel` partition use `sinfo --partition accel`
    -  You can try `man sinfo` for more options

8. What does the `tmpfs` stand for/mean in the command `df -Th  | grep -v tmpfs`?
    - `tmpfs` represents a temporary filesystem, so we can interact with these types of filesystems as if it was a filesystem, but all the data is only stored in RAM
        - Follow up question: what is a filesystem?
            - All computers have different parts, the CPU (Central Processing Unit) is responsible for performing the computation. In this context the filesystem is the place where we can store data permanently. We call it a filesystem for historical reasons and the computer can have multiple filesystems with different properties and attributes. E.g. On Saga all compute nodes have access to the global filesystem, but they also have a local fast filesystem that we can utilize for higher performance.
            - Another way to look at it is as a collection of hardrives that are made to function as a unit and made accessible to store data.

9. So the command `df -Th`  gets us the (d)isk (f)ree space that includes file (T)ypes in a (h)uman readable format. Then we use `| grep -v tmpfs` This is piped (|) into grep which looks for instances where we in(v)ert the search for tmpfs. So the full command `df -Th  | grep -v tmpfs` is showing us times where it does not match the temporary file system match?
    - That is indeed correct! Your analysis is perfect, and to verify that you can use `man df` search for the relevant flags and then correlate that with `man grep`, but I guess this might be what you did for your initial analysis :smile:
        - It is, but came from the course. The documentation says "You can also explore the available filesystems..." then gives that command, which seemingly is a way to not explore the file systems
            - to also give some context about where the teaching material comes from: it comes from an attempt to be as generic as possible and over time we have put in specifics about our cluster. there are probably places in the material which work on a "generic linux cluster" but maybe may not work so well on our cluster. but I am taking note of this so that we clarify this better in future
                - We could probably have said that it is a way to explore  metadata about the filesystem :thinking_face:


### Scheduling jobs

https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/13-scheduler.html

10. Are we expected to follow along and type?
    - No, just watch

11. Why do we need to specify shebang `#!/bin/bash`?
    - on our clusters it will work also without since the default shell is bash. but it is good practice to specify it so that the script also runs on systems where the default shell might not be bash (I am now typing on a computer where this is the case so scripts that explicitly specify this are easier for me to run)
    - even better than `#!/bin/bash` (assumes that bash is in `/bin`) is `#!/usr/bin/env bash` (does not assume it)
    - that shebang line specifies which interpreter to use to interpret the text/script below (just for the bigger picture)
    - Can i user csh if def
       - you can
       - if you want to interpret the script with csh instead of bash, you would use `#!/usr/bin/env csh`
       - if you want to change your default shell to csh, that is in principle possible but I strongly advice against it since a number of tools on our cluster work well with bash and less well with other shells (e.g. the module system which we will mention later). So on my own computer I use fish (a different shell) but on cluster I do not change it and use bash as default.

12. How long does it usually take to get an account accepted after applying for an account?
    - I am unsure but here we are trying to process within minutes (?) but also watching instructors type and watching others screenshare during exercise is still useful.
        - Yes, it is still useful.
    - Could you contact the host by email/direct message so that I can speed up the process?


### Poll: I have found my project:

vote by adding a "o":
- Yes: oooooooooooooooo
- No : o
- No, I don't have a user account: o
- Yes, but I get a warning that setting locale failed, and that I should check my  locale settings + falling back to the standard locale ('C') 0
    - [Solution to solve warning about locale failed](https://documentation.sigma2.no/training/material/2022-spring/q_and_a-day1.html):

    ```bash
    export LC_ALL="en_US.UTF-8"
    export LANGUAGE="en_US"
    export LC_CTYPE="en_US.UTF-8"
    export LANG="en_US.UTF-8"
    ```
    - and if this solved it, you can add that to `.bashrc` on the so that you don't have to retype it

13. I have tried adding the above to my .bashrc and it is frozen with the following warning "-bash: warning: setlocale: LC_CTYPE: cannot change locale (UTF-8): No such file or directory"
       - confirmed. just a sec looking for a solution
           - I fixed it above, it should have been `export LC_CTYPE="en_US.UTF-8"`
    - thank you! how do I update the .bashrc now it is stuck? I tried closing the terminal and
        -  You can reload `.bashrc` with `source ~/.bashrc`
    - but it is stuck, and when I close the terminal and
 the
         - working on it (I am recreating what you did so that I can suggest how to get out of it)
         - I put that into my bashrc, I got the warning but it doesn't get stuck so not sure what to suggest here. If you like, you can send an email to ... with your username and I remove it from your .bashrc with administrator powers
         - will do
            - cheers. fixing ...
            - it is hopefully fixed now
            - yes thanks!
               - great :-)

14. If this document (HackMD) behaves weird for you (e.g. your text disappears suddenly) can you please let us know here? I see sometimes text appear and disappear and my own text disappear and unsure what the reason is.
    - yes
    - yes
    - and for me
       - ok thanks for feedback. working on a solution.
       - I have archived older questions to a different document. Hopefully it will work better now with a shorter document. If not, we have a backup solution that we can switch to if this becomes too painful.

15. If you say that you want more than one core or node, do you have to specify what code should run on what core or node?
    - No Slurm will run the same commands on all the nodes (of course there are ways around this, but we will not talk about that in this course as that is *very* advanced Slurm suage)
        - So asking for more nodes will only repeat the same commands on each node? Not make the execution of the commands faster?
            - Correct, you have to support multiple nodes/cores yourself. To support multiple nodes one can use MPI while locally on a single computer (with multiple cores) one can use OpenMP or other shared memory frameworks.
            - Some programs already support this, while other times you may have to implement this support yourself

16. Is there any way to specify that you need exactly batches of 10 cores at each node for a job? Do we have any control over locality?
    - Yes! This is indeed possible, you can combine `--ntasks-per-node` with `--nodes` to achieve this behavior

17. Will there be a problem if multiple users request max number of cores? If I reserve 5 cores, is it going to be reserved for my job exclusively? or slurm does some smart CPU/core scheduling?
    - Slurm resources are always exclusive, on some systems (like Saga) Slurm can share nodes between jobs (but the cores will still be exclusively used by each Slurm job), and on other systems (like Fram) it may not behave like this and reserve nodes exclusively for each job

18. Is there a way to do in the same sbatch job something that runs on say 8 cores (that would be for instance to compile some code), then run the compiled code on 512 cores, and in the end do some post processing with 32 cores. That would be in the same job so that it can run over the week-end, or while I am away (and not have to come back to submit these jobs separately)?
    - Yes there is a way to achieve something like what you are asking for. The easiest way to do this is to use [`--dependency`](https://slurm.schedmd.com/sbatch.html#OPT_dependency) between jobs so that the jobs wait for each other. One way to achieve this is to create three different Slurm scripts, one for each step, then submit the first job and hold it (`--hold`) and then submit the second job with `--dependency` on the first job ID and then repeate for each subsequent step. Once all steps have been submitted you can release (`scontrol release <job id of first job>`) the first job and all the others should wait their turn (or never run if the first job fails)
        - OK, I will try that, thanks
    - I will also note that the first two steps can probably be combined here, the compile job is probably quick enough that you don't loose too many compute hours by compiling with 512 tasks, however, for the subsequent post-processing, using `--depdenency` is a very good thing to do.
    - `--depdendency` is also a very good technique to use when running heterogeneous jobs on different partitions (e.g. if one step requires GPUs, using `--depdendency` can be a very good way of speeding up your jobs by letting CPU only parts run on the CPU partition)

19. What is PEs when I try freepe command?
    - PE stands for processor equivalent. This used in evaluating how many cpu hours it costs. simply if you run a job using one core for one hour , that is 1 cpu hour. But is you use *"a lot of memory"* then you will be charged for more than 1 cpu hour.  PE is a value calculated using this information. The formula used is (where KMEM is a constant deciding what is the value for *"a lot of memory"*  )

```
PE= NumCPUs
if(MinMemoryCPU>KMEM){
    PE=PE*(MinMemoryCPU/KMEM)
}
PE_hours = $PE * TimeLimit / 3600
```

20. If you want to run a python script from a batch script, what will be the working directory inside the python script?
    - It will be the same directory as where you ran the `sbatch` command

21. When is it helpful to ask for more than one node (or tasks per node or cpus per task or threads per core)?
    - If you have a job that utilize MPI using more than one node can be a way to parallelize over even more cores than what a single node can offer
    - What sort of software are you usually using? E.g. Python, R, a specific software like AlphaFold or other? So that I can give a more in-depth answer :slightly_smiling_face:
        - Usually python with anaconda
            - Excellent, then on a single node you could parallelize with the [`multiprocessing` module](https://docs.python.org/3/library/multiprocessing.html)
            - To parallelize over multiple nodes you could use either [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) or simpler use the [`dask` framework](https://www.dask.org/) (We highly recommend this last one to automatically scale from a single core to a whole node to multiple nodes)
    - Thank you! How does this relate to parallelizing (if that is what it is called) using `#SBATCH --array=x-y`?
        - Slurm job array support is a different mechanism that is used to run many similar Slurm jobs in one go. Slurm job arrays are more suited for processing many files independently, e.g. if you had 10 files that all need the same processing, but there is no overlap (so usually you would do it in a `for` loop and save the output to independent files).

22. How can one monitor how much memory and cpu capacity is being use while the project is running? and how can one benchmark the performance?
    - you could try the command: `sstat --jobs=your_job-id --format=jobid,cputime,maxrss,ntasks`
    - About the performance: first you need to look how your code scales over a certain nbr of cores.
    - we have a tutorial: https://documentation.sigma2.no/jobs/choosing-memory-settings.html
    - see also next question

23. Sometimes we might run something on hpc but not use all the capacities available, i mean the code is not optimized to be run on multiple cores and higher memory capacity, now the question is how can one track this and see how much of the requested capacity is used up while the code is running.
    - we will talk about it at the Nov course but this is hopefully helpful:
      - [How to chose number of cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)
      - [How to estimate memory usage](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)
    - in short, one needs to monitor and calibrate and I recommend to start small and grow the computation to more cores until I can see that it does not make sense to go further
    - You could also do code-profiling (e.g. gprof) to see where the code spent the majority of the time. This helps for any optimization effort.
    - And if the code is large and you need assistance about optimization, you could look here at the [advance user support service](https://www.sigma2.no/advanced-user-support).


### Exercise: [Scheduling Slurm jobs](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/13-scheduler.html#customising-a-job)

- Use the training project:`#SBATCH --account=nn9984k`


### [Transferring files](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/15-transferring-files.html)

24. Is the hackmd (this document) now behaving better than earlier today?
   - better: o
   - same: 0
   - worse:

25. Are there any file manipulations (from this section) you can do in terminal but can't do in GUIs like WinSCP or FileZilla?
    - the copying (scp) they all do
    - later in the episode there are examples on how to create an archive out of many files (to copy one big file, fast, instead of thousands of tiny files, slow) and I am not sure whether this is available on those tools
    - something we will look at tomorrow: https://documentation.sigma2.no/computing/responsible-use.html#transferring-data
      - so there are even more ways to transfer data but `scp` is the most standard way on linux machines

26. If the data is stored on Educloud, for doing analysis on fox no file transfer is needed? that means files on Educloud are visible by fox, but is that the case for other sigma2 services or is it limited to fox?
     - how do you access EduCloud from Fox? via scp? or is it mounted as a "folder"? (I am asking because I have never been on those systems but knowing how you access I can comment on access from other clusters)
    - i don't exacly know how to access it, but i assumed that the data stored on educloud is already on the same server where fox operates on.
       - I will ask colleagues who work with that system to comment

27. Is filezilla equally fasst as scp?
     - I assume so because under the hood, also these tools use scp. But as the instructor says, it's better one or few big files than many small files. So archive/pack the tiny files first before sending them over the network.
     - also see: https://documentation.sigma2.no/computing/responsible-use.html#transferring-data


28. Imagine that the code to be run on Saga has some dependencies on some other libraries (for example nibabel package from pip python). Who will install additional dependencies in Saga? Will I get permission to install such things? What if my code/script is trying to do the installation by doing things like git, pip install or nuget-install? Will my script have access to the WWW?
    - first question: yes you can install. example: https://documentation.sigma2.no/software/userinstallsw/python.html
    - also EasyBuild and containers are available to users
    - second question: yes the script should have access to network. I write should and not will because it goes through a proxy which should let through "typical" things like git clone etc. So if it does not work, please reach out to us.
    - More on software tomorrow

29. Where is the best place for storing large data that are going to be analyzed? Is it somewhere on home directory? or somewhere else?
     - You could consider NIRD storage (see here applying https://www.sigma2.no/data-storage), which is mounted to the national clusters.
     - how large is "large"? and assuming that this is data that is not finished yet and not archived yet but still living and growi
    - in the order of a few TB
    - New NIRD platform has an initial capacity of 32 TB with a possibility of extending it up to 70 TB (future).
    - how do you access the project folder? where is this directory?
      - https://documentation.sigma2.no/files_storage/clusters.html
      - https://documentation.sigma2.no/files_storage/quota.html
    - in project folder but not in home. project folder is less limited in terms of quota. if the project folder is not large enough, please reach out to us.

30. How to transfer from repository (e.g NeLs) to HPC using scp
    - what is NeLs? does it provide command line/ shell access?
    - https://nels.bioinfo.no/
    - In NeLS you can find ssh connections detail under My Profile. Then you can ssh to  NeLS and use scp there to transfer data to for instance Saga.

31. My script after doing a lot of computations, generates a GUI window, where the user can do live interaction with the processed 3D dataset . Can I do such things with Saga? Is there a way to get a GUI instead of a terminal? If No, how can I do such tasks(interactive visualization)?
    - first step: I would separate the computation from the interaction. then the interaction work can either happen on the laptop or on the remote system. it is possible to log in graphically to the clusters also (use mobaxterm if you are on windows) if the graphical analysis is not too heavy to not block the login node for others.

32. Xconnections supported?
    - yes

33. Who is helping out answering all these questions? You are doing a great job!
    - thanks! couple of people :-) (the staff)


## Feedback about today

- one thing that was good and we should keep
  - Those who manage this hackmd file, i am impressed, this is very pro! the way the questions are covered on the go is very interesting.
    - we will try to make this more robust. it has been a bit jumpy but we have alternatives
  - would be nice if these files are aviabable for some time after the course.
     - absolutely: https://documentation.sigma2.no/training/material.html

- name one thing we should change/remove/improve for tomorrow and next time
  - the lectures can be a bit to more the point and consise, but great anyway.
