---
orphan: true
---

(training-2023-autumn-onboarding-notes-day0)=
# Questions and answers from HPC on-boarding Oct 2023, Day 1 (Oct 4)

```{contents} Table of Contents
```
## Icebreaker 

- Have you experienced any computational analysis/tasks that was too big or too long for laptop or the desktop? elaborate

    - Once I had to perfome an analysis that took 60 hours straight to complete. I could not keep the laptop on continuously for that long. 
    - my climate simulations need more resources than my laptop have
    - Numerical ocean modelling: restricted to very small simulations on a laptop
    - Whole genome sequencing and functional annotaion. too big for online tools
    - I need a cloud solution to run scripts for browser-based apps, which can also include online learning.


## Why use a Cluster?
- https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/11-hpc-intro.html

1. Will the exercies and the coursepage still be there after the course is done?
      - Yes, they will stay.

2. is data transfered to server like sigma or stay on local computer ? if yes, it means need some time to transfer data to server ? 
      - when you transfer the data, you are copying to the cluster, so it will stay on your local computer until you delete it on your local system. The time to finish the transfer depend on many factors, like size of the data, filesystem etc. We will discuss this later today


### Poll: Located ssh (type ` ssh -V`, and I see a version printed, not an error)
- Yes ooo0oooooooooooooooo
- No:o 
- 


3. I have version 3.0.1 from 2021. Shoudl i update ssh?
   - No, you don't need to update.

4. I got permission denied after writing my password, what should I do?
   - Do you have an account? can you contact the host(dhanya) on zoom with your username
   - Dhanya is checking
   - Resolved

5. Is key authentication disabled on Saga?
   - it is not disabled (I use it)
   - I tried adding my public key to `~/ssh/authorized_keys`, but it did not work
       - it is in `~/.ssh/authorized_keys` and make sure that file is only readable and writable by you. you can make sure it has the right permissions with: `$ chmod 600 ~/.ssh/authorized_keys` - does it work now?
   - Now it works :-)
        - great!

6. Is it possible to manually choose login nodes, sometimes I run 'nohup' command in one login node but when I want to kill it I cannot kill it in other nodes
    - Yes; you may log in using the direct address of the server; Saga has 5 login nodes, named login-1.saga.sigma2.no and so fort until login-5... For login 2 the address would be yourusername@login-2.saga.sigma2.no. But as Sabry said, you are not guaranteed that the login server have capacity to do your job if it is heavy loaded. The distribution on login nodes follows a round-robin principle to distribute load as evenly as possible (in theory).



### Poll: Login into Saga (type `ssh username@saga.sigma2.no`)

  * I managed to login oooooooooooooooooooo
  * I didn't manage: 


7. How can you make the file hidden?
   - files and directories that start with a dot are "hidden" when you type `ls`. Example for hidden files would be: `.somefile` or `.anotherdirectory`. You can list them with `ls -a`
   - what command make a file hidden
      - `$ mv somefile .somefile` will rename the file and by adding the dot in front we made it "hidden" (but it is not really hidden, only not shown in a `ls` command)
          - thanks
   - historical anecdote: the fact that files and folders starting with dot became "hidden" happened by mistake in the early days of UNIX. but the creators of UNIX figured that it was useful so they did not fix the mistake and kept this as useful feature to be able to hide some files from a default `ls`.

8. is saga already installed all software or user need to install like some python libary etc?
   - most of them are installed. We will talk about it tomorrow.

9. if the jobs takes more than 7 days for computing, what will happen in such case?
   - you won't be able to submit a job for longer than 7 days. and when reaching the maximum time, the scheduler will stop the job.
   - in practice this means that we need to organize our calculations into portions that fit into the time. it might require code optimization or to chop the problem into smaller problem sizes. the reason why we don't allow extremely long calculations: 1) it would make scheduling harder for other users and 2) it would make it more difficult for staff to apply upgrades on the cluster.

10. Is machine learning model data need to upload on sage server ?
    - if you wish to simulate the model with the data on the cluster, yes. But in this course we will not go in detail about machine learning model

11. I wrote info and now i cant get out of infopage, please help 
    - type: "q"
       -  I ended up restarting my terminal, but this is probably a better way
           - yes :-) many of these tools can be exited by typing q. but sometimes I don't know either and have to stop the terminal. Typical trick is ctrl-c or ctrl-d.

12. What kind of GPU are availables? and how I can check it ? can we choose specific GPU?
       - Saga has Nvidia P100 GPUs and Betzy has A100 GPU. LUMI-G has AMD MI250X GPUs. You could run the command `nvidia-smi` for NVIDIA GPUs and `rocm-smi` for AMD GPUs. More details are provided here https://documentation.sigma2.no/code_development/guides/gpu_usage.html#gpuusage

13. I tried downloading a python package with pip, but it didn't work. Are there limitations on what packages you can install? 
     - no there are no limitations on what python package to install. but the preferred way is to either install into a virtual environment or a conda environment. maybe you accidentally tried to do a system-wide install and that has to fail otherwise users could change the system for other users.
         - Ah ok! How can i know if i did a system-wide install? I naively wrote 'pip install pytorch' right after a log-in to the saga cluster.
             - it probably complained about write permissions. But a simple `pip install something` without being in a virtual environment or a conda environment will fail on the cluster because it will try to install it system-wide.
             - hints on how to do it instead: https://documentation.sigma2.no/software/userinstallsw.html

14. Can you point us to functions that work like sinfo and nproc etc on mac?
     - (I hope somebody on macOS can answer; I am Linux user and I simply don't know)
     - I don't know the command, but you can find it with some clicks on mac, 
     - Click "About this mac" > "System Report" > "Hardware" The number of CPU Cores should be listed in the Hardware overview, to the right of "Total number of Cores"
     - nproc -> MAC command is `sysctl -n hw.logicalcpu` (This returns logical CPUs) To obtain physical CPUs you may run `sysctl -n hw.physicalcpu`
15. accel        up 14-00:00:0      1  down* c7-4? is it mean the server is down?
      - it means, the node c7-4 is down, not the server

16. Is it only accel partition that has GPUs? What are other partitions  suited for?
      - `accel`  partition is for jobs that needs GPUs. More info on partition [here](https://documentation.sigma2.no/jobs/choosing_job_types.html#job-types)
      - Thanks :3 

17. Will we learn how to use conda with hpc?
    - not in this course but we have good material here:
       - https://documentation.sigma2.no/software/userinstallsw/conda.html
       - https://documentation.sigma2.no/software/userinstallsw.html

18. shh to specific node does not allow, it shows following message. Do we need to have active job on node to login?
```
ssh c7-3
Access denied by pam_slurm_adopt: you have no active jobs on this node
Connection closed by 10.31.7.3 port 22
```
     - you can only log-in to compute nodes where you have a running job
     

19. I dont have access to the right project 
      - Can you tell me(dhanya) your username on zoom ?
      - Checking
      - Resolved

20. I am wondering if you could show us how to submit a python based code/program to run on a compute node?
      - now demonstrated but please ask more if you would like to know about something more specifically
      - You could run an interactive job, which makes a compute node available for your job. This can be done on Saga running this: `salloc -A nnxxxxx -t 00:05:00 --qos=devel --nodes 1 --ntasks 1 --mem 512M`. This will make one compute node available for 5 min. and then you can run your python script (python script.py)
      
21. python code need to  write into nano file ? can we just call our python script directly without use nano etc.
      - you have at least two options:
          - a run script or job script which on top specifies what it needs from SLURM with`#SBATCH` directives and then calls python (this is what is demonstrated)
          - you can also `sbatch` a python script directly and add the `#SBATCH` directives on top of the python script. In other words the job script or run script does not have to be a bash shell script. it can be written in any language that uses `#` as comment symbol 
      - did this clarify your question?
      - You can call python scripts. The reason Sabry used nano was that he did not have any script ready, so he created python script with only a print command. Nano is just a editor to create the file. You could crete or download files instead using other methids. 

22. how can we use –partition option? and its purpose
     - example for how and use case (in this case running a job on the partition dedicated for calculations that require lots of memory): https://documentation.sigma2.no/jobs/job_scripts/saga_job_scripts.html#bigmem
     - And here is a description of different partitions https://documentation.sigma2.no/jobs/choosing_job_types.html
     - Thank you

23. Does the order of the #SBATCH inputs within the script matter? 
     - No it does not matter.
     - what matters is that all `#SBATCH` directives come before the first shell command. It will not understand the script if sbatch commands come after shell commands.
        - Understood. Thanks!

24. How do I know the memory per node? 
     - here is an overview: https://documentation.sigma2.no/hpc_machines/saga.html
     - If you want to use sinfo, folowing are two example to find out howmuch memory(in megabytes) is there on the node c1-1
      ```
      sabryr@SAGA 04-10-2023]$ sinfo -o "%m" -n c1-1
      MEMORY
      182784

      ```
      ```
      sabryr@SAGA 04-10-2023]$ freepe | grep -w c1-1
      c1-1    8 of 40 cores free,  24432 of 182784 MB free,  5.3 PEs free (MIXED)

      ```
      - You could also run `cat /proc/meminfo` to get the total physical memory of a system and Additional info.

         - Thanks!

25. I have made a nano file called test.sh I want to run, but only get this output: `sbatch  --account=<nn9970k> --mem=1G --time=01:00 test.sh`
     - bash: nn9970k: No such file or directory
     - Remove the angle brakets in account name, you have ```--account=<nn9970k>```, what you should use is ```--account=nn9970k ```
    
26. While running sbatch I got this error message 'This does not look like a btach script,. The forst line must start with #! followed by the path to an interperter'. I didn't quite understand the message.
     - the first line of your bash script should be `#!/bin/bash` or `#!/usr/bin/env bash`
     - this line tells the shell how to interpret the script. in this case it tells the shell that it should interpret the script using the bash language. I guess in your case the first line looked differently.
     - the more big picture answer is that in linux a file is not interpreted based on a file ending or suffix. we need to tell the system how it should read and understand a script. sometimes inside a script is python code and then I would put `#!/usr/bin/env python` on top of that script


## Exercise: Scheduling Jobs
- https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/13-scheduler.html#exercise
### Status
- I am done : ooo00oooooo
- I need more time:
- I am confused: 

## Transfering files
- https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/15-transferring-files.html

27. During transfering files, do we need to speciy login host or it is optional?
     - it is optional. unless the transfer takes many hours or even days. then it might be better to specify it to make sure that some automatic DNS (domain name server) "rotation" does not break the connection after few hours and interrupts your transfer. for most transfers I don't specify the actual login node.

28. Please explain how to set path . 
     - do you here refer to the "Replace /path/to/local/file" in the instructions?
       - yes
          - in this case, on saga, type `pwd` - it will "print working directory" and tell you the path that you are on. it will look like `/cluster/home/SOMEBODY`


29. Why is rsync recommended over scp? 
     - it is faster and safer. faster because it only copies file changes and does not copy files which haven't changed. safer because you can ask it to not overwrite existing files on the target machine.
     - great, thanks!
        - it was also suprising for me to find out that when transferring very many very tiny files, rsync was orders of magnitude faster. when trasferring few files it does not matter which of the two.
        - Thanks for clarification. useful info
     - another benefit: if a connection is interrupted, rsync is designed to be able to pick up where it left off.

30. Why is rsync recomended over programs like filezilla? 
      - see the answer to the question above, it is recommended, however users can choose there preferred tool
      - Only transferring what has changed makes rsync deploys an order of magnitude faster, being said that I don't have experience with filezilla :) (DP)

31. What are the flags?
     - are way to say options and pass arguments to the commands you are in. you can find the flags associated in man pages/help  `rsync --help`

32. rsync is not recognized as an internal or external command in my command prompt?
     - you can try it rsync on saga
     - yes, but if i want to transfer from my local computer? do i need two terminals to do this?
     - which operating system are you in?
     - windows: Which version?
     - Windows 10
         - Please look here: https://www.ubackup.com/windows-10/rsync-windows-10-1021.html
     -  Please see the doc: https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/15-transferring-files.html#working-with-files-generated-in-different-environments
     -  This is a bit easier with visual studio code. Please consider this solution. 
    
33. [user@laptop ~]$ rsync -avz /path/to/local/file username@login-1.fram.sigma2.no:/path/to/remote/directory   Please explain it path means whole folder path or what? local name o0
      - Above `/path/to/local/` means the path in your laptop where you have the file that need to transfer , and `/path/to/remote/directory` it is the path in your remote cluster, here on `saga`
       

34. For me it would be handy to have a RStudio equivalent in Saga. It is suggested that VSCode could be a good option. I read somewhere that I would need some software for this to work in a best possible way (vscode-R, vscode-r-lsp, languageserver, radian). Does anybody have experience with this or any suggestions?
    - Please read more about this exact question here: https://documentation.sigma2.no/code_development/guides/vs_code/connect_to_server.html 
    - We do have rstudio in Saga, but not in the way you expect it. However, if you want to use the remote solution for RStudio - please check this page; https://documentation.sigma2.no/nird_toolkit/overview.html and consider this course: https://documentation.sigma2.no/training/events/2023-10-nird-toolkit.html
     

## Feedback of the day

- What was good today
    - organized and very easy to ask questions. Also very fast feedback.
    - The explanation of the scheduler with the restaurant is a great analogy that helped me to understand the concept very clearly. I have been using different HPC resources for years without knowing how exacly this SLURM works so thank you very much! Also the map of where we are when we login and where our jobs run is clear for me now. I appreciate very much this course, thanks to all the people who made this possible!
    - thank you not only for valuable knowledge, but also for the friendly atmosphere :)
    - Thanks to the organizers!
    - Thank you! Great session
- What are the things to improve
    - It is important to point out several times that all we are learning in this course could be applied to other HPC, also out of Norway. It is useful information for all our research careers. Thanks!
    - Might be out of the scope of this course but it would be very interesting to have a brief walk-through of setting up the environment for scientific programming in VSCode but in conjunction with remote development on HPC (e.g. Enabling X11 Forwarding to plot, Jupyter VSCode extension ... etc.) 

 

