(training-2021-autumn-notes-day1)=

# Questions, answers, and notes from day 1


## Ice breaker

- How do you connect to NRIS HPC systems (SAGA, FRAM, Betzy)
    - Detailed login instructions are under the "links" heading above
    - ssh username@saga.sigma2.no
    - I have never done it yet
    - Windows program (Putty) +
    - Windows terminal emulator
        - Win-bash
        - Git-Scm
    - Unix terminal (Ubuntu, Mac..) ++
    - VNC
    - Visual Studio Code


## Links

- [HPC intro](https://sabryr.github.io/hpc-intro/13-scheduler/index.html)
    - Two main reasons for HPC: (1) solving problems faster, (2) solving larger problems
- [HPC login help](https://documentation.sigma2.no/getting_started/create_ssh_keys.html)
    - [VNC](https://documentation.sigma2.no/getting_started/remote-desktop.html?highlight=vnc)
    - [Some hints on logging in from Windows machines](https://wiki.uib.no/hpcdoc/index.php/HPC_and_NIRD_toolkit_course_fall_2020#Windows_2)
- [SAGA specs](https://documentation.sigma2.no/hpc_machines/saga.html)
- [SAGA job types](ttps://documentation.sigma2.no/jobs/job_types/saga_job_types.html)
- [Interactive jobs](https://documentation.sigma2.no/jobs/interactive_jobs.html?highlight=salloc#requesting-an-interactive-job)
- [Examples used](https://github.com/Sabryr/HPC_teaching_scripts)
- [MPI on NRIS](https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html)
- [HPC on webrowser, Lifeportal](https://lifeportal.saga.sigma2.no)
    - <https://lifeportal.saga.sigma2.no>
    - Demos
        - Upload data
        - Run a simple job
        - Share history
        - Extract workflow


## Questions and answers

- `pwd` prints you the current directory

- please clarify what one should do on a login node
    - login nodes are the entry point for all users
    - they are shared by many users (meaning many users are using these at the same time)
    - also for code development.
    - therefore login nodes shall only be used for lightweight work (transferring data, organising files/directories, creating jobs, managing jobs, ...)
    - some very little work (quick compiling, a few seconds test runs) may be ok too, but if one is unsure how intrusive some work would be it is better to launch an interactive job on the cluster
      - if the compilation needs 8 cores and takes 2 hours, I would also submit the compilation to a compute node

- is the home folder `/cluster/home/somebody` accessible from the login node?
  - this folder is visible both from login and compute nodes
  - on our HPC systems we have one parallel file system that is shared across all nodes, meaning all nodes have access to it
      - the home folder (`$HOME`), project directories (`/cluster/project/nnABCDk`), folder for temporary data (`$USERWORK` pointing to `/cluster/work/users/$USER`) and more all reside on the parallel file system

- commands used by presenter to show which partitions are available and how busy some nodes are:
  - `freepe | head -n5`
  - `sinfo  --node c3-29`
  - You can also view system load on the web: <https://www.sigma2.no/hardware-status>

- where is the example file?
    - You can copy the following:
```bash
#!/usr/bin/env bash

echo -n "This script is running on "
hostname
```

- how to list available accounts
  - `projects`
  - `cost`

- how to list the info after using `squeue -u username`?
  - can you please clarify? not sure what you mean exactly or what you are after
      - S showed that we can see the status of our job to run, but then he also showed on which node it ran
        - while it's running: `squeue -u $USER`
        - after it finished if you know its number: `scontrol show jobid -dd 12345` (replace 12345 by its number)
        - :+1:

- I have access to Fram but cannot run the same thing
   - What was wrong?
   sbatch: error: Memory specification only allowed for bigmem jobs
   sbatch: error: Batch job submission failed: Memory required by task is not available
       - What is the memory you used? you should reduce it.
   - 1G
   -try 256M.
   - regular jobs (e.i. not bigmem) on Fram are only handed out as _full_ compute nodes, so the memory option is not necessary/allowed
   - I got the same thing with 256M, so I think I cannot run this sort of jobs on Fram?
   - can you copy it here to see what is wrong?
   -
   - On Fram, you only need to specify memory requirements if you want to run a bigmem job (job on the bigmem partition). For other jobs, please remove/comment the memory requirement.
   - Try without specifying memory (see also duplicate next question)
   - it works now, Thanks

- I do not have access to Saga. Can I do the same things on Fram?
    - yes
    - Although you cannot specify `--mem`, just remove this flag and it should run

- Is VIM the best editor? Yes. No. :)
    - The best editor is the editor you are most comfortable with.
    - Many people use emacs, vi(m) or nano because they can be run inside the terminal
    - Some editors you can also run on your laptop and they can take care of the connection to the cluster

- Should we download and upload the rest of the files on GitHub to Saga?
    - Download only what is needed for the course.
    - You can clone the repository to your local machine and use `scp` to copy the files to Saga
    - A convenient way to get files directly from github directly to saga, without going through your computer:
      - select the file on github, e.g.: <https://github.com/Sabryr/HPC_teaching_scripts/blob/master/example-job.sh>
      - click on "raw", you land here: <https://raw.githubusercontent.com/Sabryr/HPC_teaching_scripts/master/example-job.sh>
      - now you can download directly with `wget`, e.g.: `wget https://raw.githubusercontent.com/Sabryr/HPC_teaching_scripts/master/example-job.sh` - this will download the file and create the file in your current folder (on saga or wherever)

- Is there an easy explanation of the difference between cores, nodes and pizzaboxes?
    - pizzabox = every box in the rack (the hardware)
    - node = a computer unit with CPUs and memory (if you prefer, a name for every pizzabox)
    - cores = processors in each node, to compute a task
    - beware of the term "CPU", it can have many different meanings. In hardware specs, a CPU typically has several cores/threads, but in Slurm "cpu" means core
    - Cores come in different flavors, it will  be covered tomorrow.

- Getting error "can't write on swap file" when creating example.sh by vim
   - I would check whether you are in the right folder (have the right permissions to write there) but also check `dusage` that you are not out of file/space quota.

- What is the difference between `srun` and `salloc`?
    - `srun` runs a script and is also used inside Slurm scripts
    - `salloc` allocates interactive time on a compute node
        - Confusion might arise from the fact that `srun` can also be used to get interactive access
        - <https://documentation.sigma2.no/jobs/interactive_jobs.html>
        - What about sbatch vs srun?
            - `sbatch` runs a Slurm script which can contain `#SBATCH` comments to configure the job
            - `srun` is mostly intended to run a single command within Slurm
            - There are more subtle differences between them, but we should not confuse that here and now

- Does X11 forwarding work for interactive jobs?
  - yes, more info here: <https://documentation.sigma2.no/jobs/interactive_jobs.html?highlight=interactive#graphical-user-interface-in-interactive-jobs>

- Is `salloc` a better way to use than `screen`?
    - These commands do not do the same things, `screen` is a terminal multiplexer which allows you to have multiple "terminals" through the same SSH connection
    - `salloc` allocates an interactive session on a compute node
    - Does that mean if I run an analysis on the HPC I should use `salloc`?
        - You should ideally create a [Slurm script](https://documentation.sigma2.no/jobs/job_scripts.html) and run it with `sbatch`
        - `salloc` is a good way to interactively test on the compute nodes, maybe you need to compile something or need to test how quickly your program runs for 5 minutes

- Is module load loading on the login node?
    - Yes, when you run `module load` it will load the module on the login node
    - However, when you use `srun` or `salloc` modules loaded on the login node are also loaded
    - For Slurm scripts modules loaded on the login node are _not_ loaded and you need to have `module load <module>` inside the Slurm script

- How many cores a node can have?
    - [Details](https://documentation.sigma2.no/hpc_machines/hardware_overview.html#comparison-between-current-hardware) from the official documentation
    - Tricky question, what is a core ? Covered tomorrow. The command lscpu report virtual (aka logical) cores also. This is often referred to as SMT-2, common on x86-64 processors. Will go through this tomorrow.
    - for saga [here](https://documentation.sigma2.no/hpc_machines/saga.html?highlight=login%20nodes)
    - Saga: 24/40/52/64
    - Fram: 32
    - Betzy: 128
    - the trend seems to be that nodes (also laptops) have more and more cores so we better make sure that research codes can make use of those

- What are Slurm scripts?
    - Slurm scripts are scripts/recipes which contain the steps to run but also contain directives which ask for a certain number of cores and memory that are executed by `sbatch` we will talk more about these scripts later
        - A Slurm script can be written in `bash`, `Python` or similar scripting languages that use `#` for comments
    - You can read more about [Slurm scripts here](https://documentation.sigma2.no/jobs/job_scripts.html)
    - Shortly - these are scripts which inform the cluster what command(s) shall be executed and what resources shall be used when it is being executed (memory, time, etc)

- If the module we are intested is not in the module list, how is it posssible to call it?
    - The easiest way to proceed is to contact us at support@nris.no and we will help you with the specific software you require
    - but just to clarify: you meant that you would like a module and cannot find it in `module avail`?
        - yes that was what I meant, if it is not in the module avail - list?
            - Contact us and we can see if it possible to install the software or help you install the software

- Is there some difference between module purge and module restore?
    - `module purge` will unload all loaded modules (that are not marked "sticky"). `module restore` will "restore" a previous setting. By default `module restore` will do the same thing as `module purge`, but you can also save your current settings first with `module save <some-name>` and then later restore the same settings with `module restore <some-name>`.

- what does it mean for module load and module avail?
  - module load activates the environment of the module and makes it visible so that you can access its binaries and libraries
  - `module avail somename` will list you all modules that contain "somename" but it will not load them, only show you which ones exist

- Can we open the graphic user interface? like fluent?
    - X11 fowarding ? ssh -Y (please use -X if possible, Y is less safe, but works more often).See the documentation <https://documentation.sigma2.no/getting_help/faq.html?highlight=graphical%20user%20interface#graphical-interfaces>
    - VNC (remote desktop)
        - I have already installed Tiger VNC and use it to submit my tasks on Fram. I am wondering how I can open Fluent there by GUI?
- can we have conda on saga and have Env for one user
    - first question: yes. Conda is available on Saga. more info here: <https://documentation.sigma2.no/software/userinstallsw/python.html?highlight=conda#anaconda-miniconda-conda>
    - second question: not sure what precisely is meant
    - If you by "Env for one user" mean a unique environment for only your user, then yes.
      - and that's a good idea: unique environment for a user. I would even try to create a separate environment for each project (since projects can have different dependencies)

- scp vs rsync - any difference on the usage or just a prefernce?
    - Rsync can pick up if stopped, it normally for larger transfers. Both uses ssh for transport.

- Can I put a scp into a bash script before running a code?
    - You can do that if necessary, however, for better performance we recommend that you download data to Saga before you start
    - You can download large files to `$USERWORK` and load from there when running code
    - The other problem that you might have is how to authenticate without "being there" to type the passphrase. This may encourage to use empty passphrases which is not a good idea in general.
        - even if I have setup a key?
           - yes. also for ssh keys it is generally not a good idea to not set a passphrase because if I get hold of your private key, I can then log in as you. A passphrase prevents this by encrypting the private key. And normally you only need to type the passphrase once a day so it's not a huge burden.
           - Oh I meant if I have setup a ssh key (not empty), do I then need to authenticate if I scp inside a bash script?
             - ah now I get it, sorry: yes. the script will probably fail if it runs at 2 a.m. and you are not there to type the passphrase and it cannot find the decrypted key in its memory. one can give the unlocked private key probably a longer lifetime and one probably can use ssh agent to remember it for a while but I anticipate this to be difficult and not practical.
             - depending on the use case one could use some authentication tokens. so there are situations where one wants unsupervised authentication and it can be solved.

- How about the support for using ftp or other related softwares?
    - ftp is not secure, hence not allowed. Use scp or rsync.
    - not secure because data travels in plain text and it is easy to intercept it

- Are you typing these commands in command prompt in your laptop?
    - Yes, he's using linux so it's in the terminal which is the same as a command prompt, just named differently
    - When I tried to do in my command prompt, it shows this error: 'scp' is not recognized as an internal or external command, operable program or batch file.
        - That's because you don't have `scp` installed on your system, you can download the install file from [here](https://winscp.net/eng/index.php?)
        - thanks but i have winscp intalled on my windows
        - do i have to go to that directory where it is installed?
        - You can try, I'm not a Windows user so if you have already installed it I don't know why it's not being recognized, please let me know how it went when you ran it from its directory
        - I stronly recommend to try <https://mobaxterm.mobatek.net/> on Windows which will give you a terminal, graphical connectivity, and also scp, all in one
        - Yup, I use mobaxterm rather than putty, but I instaled winscp because file tranfer is slow on moba.
          - ah. interesting to know.

- Why should we not use scp command on the cluster? Maybe I misunderstood, but whenever we copy file , we should use scp on our laptop?
    - You should run `scp` on your laptop when copying so that the copy operation is not interrupted if your SSH session is interrupted
    - in principle you can call it from either side but calling on the laptop side may be easier since you can authenticate to the cluster but your laptop is probably behind a firewall so you may not be able to reach it "from outside"
    - but it's no problem to scp from one cluster to another if you are on several clusters and need to copy/move files

- Do we need to rsync/scp if we do analysis on FRAM, but have the data on NIRD?
    - Yes, data on NIRD needs to be copied to the clusters to be accessible

- Is scp or rsync is installed on SAGA?
    - Yes, both `scp` and `rsync` is installed on all of our clusters
        - Saga, Fram and Betzy

- How can we precisely calculate job running time since it may terminate or reimbrushed?
    - You can't, but you can give it what you think is too much time and then note down how long it took and adjust the requested time to something just slightly above so you have some margin for next time
    - Keep in mind that asking for too much time will give you a lower priority in the queue, so try to adjust it once you know how much time is required
    - It will benefit you as well as all other users

- Can you say something on the exit code on the slurm file ? does 0:0 means job executed successfully or?
    - An exit code of `0` means that the job exited successfully
    - Within Slurm the exit code is reported as `<code>:<task>` which is used to show how each task exited, so if you run multiple tasks you can see if every task exited successfully or if any exited with a different code

- how do I exit the example-job.sh
    - Do you mean exit the file or cancel the job?
    - exit the file, im unable to come back to the commad line
       - did you open it with vim? if yes, then type ESCAPE, then type ":q!" (no saving) or ":wq" (with saving) and hit ENTER
       - I opened with vim, but the ESCAPE wq does not work, it just gets written in the file
       - try several times ESCAPE, then type ":wq" - it should work. otherwise one "brutal" way out is to close the terminal and reconnect to Saga
       - unfortunately it still doesnt work, am I supposed to write it at the top of the file or at the bottom or where? Nothing really happens when i type ESCAPE
         - we can also go to a breakout room and look at it via screenshare? when you hit escape, vim should go out of "insert mode" and then whatever you type should show up at the bottom of the screen as command to vim, not as something you type. please also double check that this is really vim and not perhaps nano.
         - when i hit escape and try to type something i just get "error not an editor command" and thats it. I just ended the session because I dont know what else I can do
             - does it show "Not an editor command:" and then something behind that that would give us clues?
             - ok, I see that session ended. Well, happy to look at it on screenshare in a breakout room or after the course. If yes, please write at the bottom (I am watching there)

- What is Slurm?
    - <https://slurm.schedmd.com/>
    - Slurm is a queuing system which is responsible to share the compute resources between all users on the cluster
    - When you ask for resources Slurm will check when and where your code should run
    - It will also do accounting so that we can monitor and share resources fairly

- Can the `sacct` command be used to show the status of all your jobs ?
    - To see the status of your running jobs you should use `squeue -u $USER` or `squeue -j <job number>` to see the status
    - `sacct` is used to see the resources accounting
    - But if I have many jobs submitted and want to see which jobs completed successfully and which ones failed, which command should I use then?
        - To view previously submitted jobs that may have completed use `sacct --start=<YYYY>-<MM>-<DD> -u $USER` this will show all jobs completed after `<YYYY>-<MM>-<DD>`
            - Thanks that's really useful!

- "sleep" command - what is it?
    - `sleep` simply halts the program and waits for `X` number of seconds
    - It's very much like a timer, I use it if I want to queue something for later like this `sleep 10; echo "it has now passed 10 seconds"`
        - It does not use CPU while it wait.
    - we only use it to simulate a running program without introducing and explaining a real program, to minimize confusion. in real life there is very rarely need to use the `sleep` command.

- At some point I heard Sabry said "if you ask more time you will be reimbursed, if the job finished before the allocated time" if that is true, so would it be wise to ask for reasonbly more time to finsh the job?
    - Ask for some margin, but keep it sensible because you get placed further back in the queue if you ask for a lot of time
    - more about that tomorrow but yes, add some margin, perhaps 15-20%

- @bast Will you cover workflow managers tomorrow? E.g. snakemake which will be queing tasks on its own?
  - no, not me. but good point. we should have added this to the material.
  - yes, it is an advanced topic.
  - we have some course material here: <https://coderefinery.github.io/reproducible-research/04-workflow-management/> but this would be for another day
  - do you know on which day this is covered? or not yet?
    - to my knowledge we did not consider teaching it this time. we should do a separate training event on this and also add this next time.
    - ah, thanks. i hoped for that b/c i was running into some problems, but this is a topic for individual user support then. i'll have a look at the material.
      - I recommend to send an email to support and we can go from there and help. We could also potentially offer extended support.

- How do I exit the `watch` command?
    - `Control-C` shall do the job :+1:
    - Please do not leave `watch -n 1` commands on forever, they will actually consume resources, specially `squeue -u uname`

- What should I do if I don't know how much time my program takes or how much memory it will use ?
  - I will talk about it tomorrow but in short: it can be difficult to say and needs to be calibrated and I recommend to "grow" the calculation from a small example towards the real example. Run a series of calculations where the problem size is increased and hopefully one can extrapolate from that. One can also start too conservative and run the big thing and then adjust but the disadvantage is that you may queue a long time.
  - In the past I have been able to estimate the usage by running locally and guessing what's required with production sized data
      - A useful tool for local use is `htop`, it lets you see RAM and CPU usage per process
        - on clusters there is `top` while the login nodes have `htop` installed
          - aha! TIL. or today I re-remembered. nice.
          - Fram and Betzy have `htop` on compute nodes, Saga does not
             - great to know! I was testing only on Saga and didn't notice. I was in fact looking for htop and wishing it was there and nice to hear that it is (on some machines).

- Is there VNC support for betzy by any chance?
    - Fram and Saga support remote desktops
        - I've used it on fram, that's why I am wondering about betzy.
    - More information [about remote desktop here](https://documentation.sigma2.no/getting_started/remote-desktop.html?highlight=vnc)

 - `python pi.py 10` What is 10 here?
     - `10` is the first argument and represents the number of random numbers to "draw" when calculating PI
     - so this is related to python script? (I think we are giving total_count here)
         - passing some values as command line arguments to the Python script is a nice way to change values without opening and editing the file every time (this can be done also in every other language)
           - a nicer way would have been to [program it to accept `python somescript.py --num-throws=10`](https://docs.python.org/3/library/argparse.html). this can be done but we wanted to keep it simple, since this is not the focus here but for user friendliness I would make it more explicit.
         - thanks

- Parallel Computing, link shared by the participant <https://hpc.llnl.gov/training/tutorials/introduction-parallel-computing-tutorial#ExamplesPI>

- General question: `##SBATCH --qos=devel`, This is a command in one of my script. why it has two hash tags, what dies it mean?
  - Slurm understands lines starting with `#SBATCH`. If you put another "#" in front of it, like in your case, Slurm will ignore that line. So here, that line will have no effect.
  - Additional explanation: in Bash, all lines with "#" are comments but Slurm still reads lines that start with "#SBATCH" but it will ignore any line completely with more than one "#".
  - But what does "qos=devel" mean?
    - it asks Slurm to put this into the development queue. This is a queue with higher priority and shorter queuing time for development and debugging only. It is not meant for any job which is not development and debugging. Motivation is to make development on the cluster even possible (without waiting for hours after every code edit). It is also OK to use it to debug your job script. Some users from time to time, willingly or accidentally, use this to run "real" jobs and then we will contact them and ask them to move jobs to the regular partition.
    - Oh, interesting to know.

- When you use srun, is there a slurm report made? Whenever I try, I do not find one slurm report. Only with sbatch.
    - With `srun` and `salloc` a report is not created
    - A Slurm output file is only created when you run a Slurm script with `sbatch`

- Does each rank run on a different node? And does the --mem flag request memory per node?
    - Each rank does not necessarily have to run on different nodes, they could, but there is no guarantee (unless you use `--ntasks-per-node`)
        - Without specifying `--ntasks-per-node`, will it be one task per node? And then one rank per node?
    - The `--mem` flag requests memory for the whole job, i.e. the sum of memory for all tasks. This has some interesting implications:
        - If you ask for `--mem=1G` and 2 tasks
        - If one tasks uses 900MB and the other task only uses 50MB then everything will workout without issue
        - `--mem` is actually per node if your job gets distributed across more nodes. Example: you ask for `--ntasks=10 --mem=10G` and your job gets distributed 1+9 tasks over two nodes, then you get 10G on _each_ node, which means that the 9 tasks on the second node will _share_ 10G. Alternatively, by requesting `--ntasks=10 --mem-per-cpu=1G` and you get the same 1+9 task distribution, you will get 1G on node 1 and 9G on node 2.

- Which username we use to login to web-based lifeportal? is it the full user@saga.sigma2.no? Why I can't login using username for saga.sigma2.no?
    - It is the saga username, did you get an error ?
    - Yes "Auto-registration failed, contact your local Galaxy administrator. Please enter an allowed domain email address for this server."
        - We need to check accounts. Will help you offline. Please send a mail to "support@metacenter.no"
        - Thanks, will do that!
        - I have the same problem: using email gives "No such user or invalid password." or using username "Auto-registration failed, contact your local Galaxy administrator. Please enter an allowed domain email address for this server."

- Can you run Busco with lifeportal?
  - If requested we shall have it

- Is lifeportal on fram?
  - It is on SAGA
    - is there any plan to have this also on other clusters?
        - Not at the moment as this is mainly for make it an easy entry point to HPC, not as a replacement for command line. If there is a need yes possible on other places.
    - so the rest of today's session is mostly relevant for Saga users, not fram users?
        - Correct.

- font size is small for me... could you increase it? and I have a 27.5 in screen...
  - thanks for feedback, I will raise this issue

- is the program only for genomics?
  - Yes. We are showing a prototype and I will comeback about including more tools

- Can you save the sbatch scripts created by the Web interface?
  - they are always created but the web frontend will point you where you can find it on the cluster

- does "working space" in lifeportal corresponds to /cluster/work/? or what?
    - It creates its own working directory, but not exactly `/cluster/work/`

- where is the link for webinterface?
    - <https://lifeportal.saga.sigma2.no/>

- Does having an accopund on SAGA grants an account on lifeportal? Or, can I use SAGA account to log on lifeportal?
  - in my understanding: yes, with your saga account you should be able to log in
      - I tried it with my SAGA credentials, but `Auto-registration failed, contact your local Galaxy administrator. Please enter an allowed domain email address for this server.`
      - Indeed my answer was wrong and during this prototype phase not all Saga accounts seem to be mapped to this service and you may need to contact support@nris.no to be added.

- Will you share the recordings?
     - they will end up linked here: <https://documentation.sigma2.no/training/material.html>. You will receive an email when it is available.

- do the files have to be decompressed to work on lifeportal? i.e. could Sabry have run fastqc on fastq.gz files?
    - Yes, fastqz.gz could have been used instead as the groomer software accepts that format.
    - so that still means that sample files have to be uploaded one by one, e.g. upload individual fastq.gz files?
    - Files can be uploaded as archives or as individual files. It depends on the software used . I.e. some software accepts archives (also compressed) some do not. At the moment if you upload an archive or compressed file it will not be defalted. But you can run a tool to deflate, which is not instlaled at the memoment. We can make it availabel if there is a request.


- Is there lessons on how to run scripts on the Galaxy Lifeportal on SAGA? So we can recreate the steps you did today to run the script.
    - <https://training.galaxyproject.org/>
    - <https://lifeportal.saga.sigma2.no/u/sabryr/h/sharednris>

- is galaxy something we (users) can create for ouselves? I mean, can we implement workflows for applications other than what available now?
    - The workflows are not more than chains of operations executed by different tools in the system. As long as we have the tools installed, we can run them separately but also combine them and produce a workflow. In turn, the framework Galaxy is quite complex to be installed by a single person. This is why we installed it for NRIS community.

- Could you please provide the fully commented version of the final MPI parallel python code? and explain how to manage tabs and indentations correctly in vim? especially useful for users who are used to prepare scripts on notepad+ and then upload them to server.

- one thing I missed (or maybe did not hear?): where and how to determine where output files and error files are produced when submitting a script? Overall, it would have been nice to have a little overview on HOW to organise yourself on a cluster: where to put scripts, where scripts outputs etc...

- Can you explain the difference between --ntasks and --nodes in #SBASH parameters


## Feedback for the day

One thing you particularly enjoyed:
- all the questions gotten take care of quickly and professionally
- easy to understand and good explained:)
- everything was explained very quickly, and the teachers were helpful
- relative easy start into the HPC system
- fixing errors (both intentional and accidental) as we went along was useful.
- agree with the person before (error fixing during the lesson)
- pace was nice.
- the content is on an appropriate level for beginners
- good content webpage with lots of info

One thing we should change/improve for next time:
- the whole thing is too fast. if you lose one point you will be totally lost.
- maybe a little bit more regarding the calculation of the resources that should be used for example job
- a little bit too fast paced especially for beginners, but perhaps I misunderstood the course, and maybe this is for people who already had some experience with HPC
- the hands on sessions where too fast, hence hard to follow
- bit fast
- perform some more testing of the demonstrations in advance, so that we can focus on the point of the demo instead of trying to follow along when solving unexpected errors
- don't use vim, it is too hard for a beginner :-P
- in particular for the terminal work, when there is a slightly complicated command output to analyze; show/tell which part that brings you to a certain conclusion
- Generally, it was a bit complicated to follow what you were saying and the course content material, as it was complicated to navigate between the hackMD (and the many questions!) and the course content page, and you were not always following the same structure as the course content. Overall, much too fast for real beginners, but appropriate content and speed for people who have already used a cluster.
- should have left the time for users to try out and do the exercises of the course page, but this makes the course longer.
- I would love if things were a little bit more structured rather than just running a script with no explanation of the code there and being surprised by the output/error messages. The documents on the github page are good, but it would be more easier if we stick to the script. Also, the topics today were not that hard, but there was a flood of commands that we needed to use.
