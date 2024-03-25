---
orphan: true
---

(training-2022-spring-notes-day1)=
# Questions and answers from HPC On-boarding May 2022, Day 1 (May 4)

```{contents} Table of Contents
```


## Icebreaker question

- Have you experienced any computational analysis/tasks that was too big or too long for laptop or the desktop?
    - During a bioinformatics anaysis analysing a single files took 1 hour on office desktop. I need to anaysis 450 of those.
    - Yes, for one project I had to process very many short independent jobs and I had only 3 days to do that and laptop/time wasn't enough to do them one after the other.
    - Analyzing 2-photon microscope video recordings that are typically 200gb in length.
    - I also had a bioinformatics problem with too many optimization runs to do it locally on my laptop
    - With atmospheric simulations it is too big for a laptop sometimes.
    - Analyzing metagenomic data and need HPC for these analyses
    - yes, used a GPU
    - Yes, running climate models
    - Yes, transcriptome mapping of RNAseq data
    - Yes with earth system model simulations and data analysis


## Why use a cluster?

[Teaching material](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/11-hpc-intro.html)

1. Now I'm confused -- Cloud vs HPC/cluster discussion -- no need to interrupt
    - Could you specify and we can optionally interrupt the presenters (or answer here)
    - Which specific part of the "HPC vs Cloud" discussion would you like us to clarify
    - Azure offers HPC instances with InfiniBand
        - [This system is actually the 10th fastest in the world!](https://top500.org/system/180024/)
    - My five cents to the issue: Imagine Cloud as a virtual shop for computers. You order 1,2 or N machines there. You receive credentials to log in and you can use them at your own discretion. They may live in different geographical locations. These machines are not connected with a high-speed connection between each other and you can not send jobs to all of them simultaneously UNLESS you specify that these machines shall be configured as an HPC-cluster and are connected between each other with a high-speed connection (e.g. Infiniband). The latter configuration increases the Cloud price solution as the machines will have to be located in a similar geographical location. In the case of Cloud you pay a) for upload and download (e.g. per MB) from/to Cloud machines and b) for the time the jobs are running (e.g. per CPU/hr). In case of an HPC cluster (e.g. SAGA), you _always_ get access to the same amount of machines which are interconnected with a high-speed connection. It is up to you how many of them you request for your calculations.


## Working on a remote HPC system

[Teaching material](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/12-cluster.html)

(add an "o" at the end of the line)

- I was able to login to the cluster: oooooooooooooo
- I was not able to log in:
- I did not try, only watching:

2. Will we have to select a partition avail (normal, bigmem,..) when running a job?
    - You normally don't need to specify, because you will get partition `normal` by default, however, if you have specific needs (e.g. a lot of memory `bigmem` or GPUs `accel`) then you will need to specify (note also that these can incur additional costs)
        - Great all clear now, thank you!

3. What should/can I do on a login node and what should/can I do on a compute node?
    - you shall put your job scripts to the login node and start them from there
    - you shall try not to run any jobs which are longer than 20-30 min on a login node
    - you may want to compile something on a login node provided this does not take more than 20-30 mins
    - compute nodes shall execute the jobs launched by the job scripts, it is quite unprobable that you have to log into the compute nodes
    - compute nodes are used for debugging, as long as important parts of the logs of the jobs land into compute nodes' scheduler logs

4. Is "head node" the same thing as login node?
    - Yes, we try to use the same terminology everywhere, but sometimes outside words sneak in

5. Is it recommended to use htop to see the availability of nodes in saga?
    - `htop` will only work on the local machine where you run it, so it will only show you the load on the login node in most cases
    - One needs to use tools such as `sinfo` to see the status of the whole cluster, this is a bit more advanced and I don't think we will cover that in this course
        - You can read more about [`sinfo` here](https://slurm.schedmd.com/sinfo.html)
            - Great, now I know what to use :)


## Scheduling jobs

- [Teaching material](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/13-scheduler.html)

(add an "o" at the end of the line)
Question to participants, how is the speed so far?
- too slow:
- too fast:
- just right: oooooooo
- too confusing:

- [Core counts on NRIS clusters](https://documentation.sigma2.no/hpc_machines/hardware_overview.html)

6. Are we sent to free nodes automatically when submitting jobs?
    - Yes! The queue system will only give you free resources, on Saga you will be the only person that has access to the given CPU cores, but other people might be using the rest of the CPU cores if you don't ask for all the cores. For Betzy you will only got free nodes

7. When is a node marked as allocated? Can you elaborate on that?
    - Before a job starts, it gets allocated a node (or number of nodes). This means that your job will run on these pasrticular compute nodes. You can check which exact nodes are being used for your job by a special command which will be given later, I suppose.
    - What is the difference between a node being busy and node being allocated? Are they the same?
       - a bit of a detour answer: ideally, nodes are busy all the time and almost never idle and as soon as one job finishes, another queuing job should start immediately. However, it is a bit more difficult because it is more likely that always only few "restaurant seats" become available but never large parts of the restaurant at the same time. So if the scheduler is too simple, then jobs that require many cores at the same time, would never start and always queue. Therefore, there are different queues and different priorities and sometimes the scheduler will have to reserve a node/processor waiting for a larger job. If I understood correctly. I am also not 100% sure about the technical distinction.
       - a node is actually allocated to a job before the job starts and is technically _not busy_ with this job at this moment, but, as said, it is most probably busy with other jobs already running there
           - Thanks!

8. Can I have a separate terminal window open that is not on Saga?
    - you can use the `terminal` in windows again and another terminal will open in your local machine
    - each new terminal will start a new session on your local computer

9. the "#" usually indicates that the line is not actually executed, how is this different for this shebang?
    - correct. you can put comments starting with "#" and they get ignored
    - however there are exceptions:
      - the so-called "shebang", `#!/bin/bash` is actually read and understood if it is on the first line
         - Is that because of the "!", or is there another reason why this is accepted?
           - yes, it tells the shell which interpreter to use to read and interpret the script/program that follows. anything following "#!" is an interpreter. The interpreter does not have to be Bash, it can for instance be Python or R or Perl or some other language.
      - `#SBATCH` commands that we will learn about to ask the scheduler for resources are read and understood by the scheduler but this is not a Unix feature but a Slurm feature

10. how do we kill a running script on saga?
    - `scancel JOB_ID`
    - is it a Slurm job script or a Bash shell script?
    - its the script that sabry made (I did it also)- which is now running?
       - the one that was started like this: `./example-job.sh`?
         - yes
           - and it never finished/returned and is hanging?
           - this is the message from terminal: "This script is running on login-3.saga.sigma2.no" is it still running? its not really doing anything
              - when it printed this, it finished. because this is the only thing we asked the script to do: to print that line and tell us on which `hostname` it was. So all is fine. It is not running anymore. You can check all your currently running processes with command `ps`.

11. I am getting the following `warning` message, when I type `projects`

    ```
    perl: warning: Setting locale failed.
    perl: warning: Please check that your locale settings:
	LANGUAGE = (unset),
	LC_ALL = (unset),
	LC_CTYPE = "UTF-8",
	LANG = "en_US.UTF-8"
    are supported and installed on your system.
    perl: warning: Falling back to the standard local      ("C").
    ```
    - Should I change some settings?
        - Yes (I guess you are using a Mac -> Yes): put this into your file (in your home directory of your mac) `.bashrc`and run `source .bashrc`
    ```
    export LC_ALL=en_US.utf8
    export LANGUAGE=en_US
    export LC_CTYPE="UTF-8"
    export LANG="en.US.utf8"
    ```

    - works great!
    - Nice to hear!!

12. the obvious question is: how do I know how much memory and time I will need? 1 min and 1 Gb seems a bit much for this also
    - very good and important question. just a sec ... looking for good links in our documentation but in short: this needs to be calibrated for each job class/type. ask for not too much but a bit more than you think (20% more?)
    - how to calibrate memory: https://documentation.sigma2.no/jobs/choosing-memory-settings.html
    - how to calibrate number of cores (we did not discuss this part yet): https://documentation.sigma2.no/jobs/choosing-number-of-cores.html
    -
    - how to calibrate time: run one example job and check how long it took before starting dozens similar jobs.
    - start with a small example and grow the job/ system size/ problem size and with this you will get a better feeling for how long the real/big job will take

13. where do I find my account number?
    - The command `projects` will list all projects (accounts) you are associated with
    - The command `cost` will also display how many cpu-hours you have available in the different projects


14. How do I know the time and memory my job will take?
    - hard to predict and needs some calibration. see also question 12. in short: run one example job and check how long it took before starting dozens similar jobs. if it timed out, ask for more. takes some calibrating. also grow the calculation from small to realistic so that it will be easier for you to predict. I often start with asking for 1 h and then go from there. For the first ever job it is also fine to ask for more. But if you ask for excessively more, say 5 days, then you may queue way too long. So it's better to start small/short and add rather than too much and remove.

15. Is there a maximum time limit?
    - Yes, it is different for each partition type. You shall refer to the SAGA/Betzy/FRAM documentation.
    - https://documentation.sigma2.no/getting_help/faq.html?highlight=walltime#jobs-submission-and-queue-system
    - The command `sinfo` that we already saw also gives some info on this. On the "normal" compute nodes on Saga the limit is 7 days (format `DAYS-HR:MIN:SEC`):
    ```
    PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    normal*      up 7-00:00:00      1  inval c10-33
    normal*      up 7-00:00:00      1  down* c5-52
    normal*      up 7-00:00:00      3  drain c1-10,c3-22,c5-39
    normal*      up 7-00:00:00      5   resv c2-[5,9,27],c3-19,c5-45
    ...
    ```

16. What happens if you give to litle time?
    - the job will be stopped by the scheduler and you will find an error message that it timed out. if you have a long running job that risks to time out, you can also contact support and ask us to extend the time. this is OK if this was an "accident" and you risk losing days of compute time. but extending jobs should not be a regular thing since it makes it harder for the scheduler to schedule (restaurant analogy: somebody reserved a table until 9 pm but stays 2 hours longer)
    - please don't count on extensions. They are resevered for special circumstances. If your job needs longer than allowed on the partition you are using, we can help you to improve your workflow by parallization or checkpointing.

17. Can we have empty lines between the `#SBATCH` commands?
    - Yes
    - But all the `#SBATCH` commands need to be put in the beginning of the script, before any actual job commands

18. how can i ask for 0,5 gb mem: comma or dot? rediculously small but just to know what the decimal character is?
    - Type `--mem=500MB`
         - yes that is a solution-is there never a situation where you want to  write a number with decimals?
         - Decimals are not allowed as a value to `--mem`

19. I was kicked out if SAGA and now when I try to access it it says "operation timed out"
    - `ssh` back into Saga does not work anymore?
      - No it just says "operation timed out"
         - is this a firewall issue? also asking NRIS colleagues
             - I was able to access at the beginning of the course
             - Try running `ssh -vvv ALL_NORMAL_OPTIONS` to get more debug information (ie, where it stops working).
             - Yes I get lots of information now. It says: "authentication provider SSH_SK_PROVIDER did not resolve; diabling". Is that the key maybe? maybe it's more efficient to solve this issue in a breakout room where you could share your screen. Yes, I can contact you. I will just quickly try to restart the machine and see if it helps. It seems to be lagging a bit as well. it may be sufficient to reconnect to your network (as a first step)
             - Now it works. Thank you so much!!

20. What does sleep 60 mean in the job script or why put it there? I missed the explanation.
    - It means : "stop execution for 60 secs but stay in the job"
    - we did this so that we have a job example which "computes" for 60 seconds. Otherwise it would finish within miliseconds. In this artificial case it does not do anything but only waits. But we do this so that we can inspect job states before it finishes.
    - `sleep N` returns after `N` seconds. `man sleep` shows a very concise description of the program.

21. Presenter mentioned job states (if I understood correctly) and we list them here: https://documentation.sigma2.no/jobs/monitoring/job_states.html#job-states
    - but maybe it was something else? availability of nodes? (I was typing and distracted)
    - yes, these are the states that you job can be in when you inspect it with `squeue`, under the `ST` column
    - but once it completes it will disappear from the list, so you will never actually see it in `CD` (completed) state

22. I am running on `betzy` and getting the following error in `sbatch`
    `sbatch: error: --nodes >= 4 required for normal jobs`
    - Betzy is meant for big jobs which require big resources. Small jobs requiring very few nodes are not allowed (welcome) there
    - You can add `#SBATCH -qos=devel`, then it starts faster and you can go down to only one node.
        - Works with the `devel` setting :)
        - Keep in mind that the `devel` QOS is intended for testing and development - not production runs.
        - all registered participants should have acces to saga and fram. You can do the excercises on Saga as well.

23. What's the difference between login node and compute node, from the perspective of hardware? e.g. login node is CPU and compute node is GPU.
    - Basically no difference from hardware point of view but it depends on the decision of the admins
    - No, the login nodes are CPU nodes
      - So compute node is GPU?
        - most compute nodes are CPU nodes but there are some GPU compute nodes also
        - the difference is in how we use login nodes and compute nodes:
          - login nodes: many users at the same time, interactively editing files, submitting and monitory jobs
          - compute nodes: few or only one user at the same time, computing, often in batch mode (not interactively; but one can also get an interactive compute node)

24. How can I cancel a job?
    - check the job number with `squeue -u yourusername`
    - then: `scancel jobnumber`
      - Thank you!!!

25. `squeue` has an option (`-i` or `--iterate`) to query slurm with some interval. Might be better resource usage than using `watch`.
    - thanks! Today I Learned.

26. How to add comments in the submission script?
    - Just use `#` before the line. _Not_ before the shebang and the `SBATCH` keyword lines though
        - For the `SBATCH` lines will adding `##` comment it out?
            - Yes
                - Thanks!

27. How do we specify the output folder for the completed job?
    - By default it is the folder where you launch the sbatch script from
        - Can we specify some other folder to redirect the output?
           - in your job script you typically have a line or lines that do the actual computation and produce output and you can redirect it then with ">" to any path/file on the system that you have access to. Side note for later: If your computation is very input/output heavy (lots of writing every few miliseconds, very many writes), then you need to also be a bit careful where you redirect to as you don't want the disk bandwidth to become the bottleneck of your calculation.
           - `#SBATCH --output=/SOMEDIR/%j.out` but mind the previous comment hereabove
               - Thanks!

28. In the 15th question, resv c2-[5,9,27],c3-19,c5-45... what c2-[5,9,27], c3-19 means specifically?
    - `cX-Y` is the name of the compute node, so if log in to one and type `hostname` it will give you this
        - is it from X to Y or X-Y together is the name?
            - `c2-5` is one node, `c2-[5,9,27]` is a shorthand for the three nodes `c2-5`, `c2-9` and `c2-27`
        - [5,9,27] are reserved nodes? the rest are not?
            - in this case all the nodes `c2-5`, `c2-9`, `c2-27`, `c3-19` and `c5-45` (so five nodes in total) are reserved
    - `resv` means that all the nodes listed are reserved for some reason (probably this course)

29. #SBATCH --reservation=nn9989k ,   #SBATCH --acount==nn9989k    both nn9989k?
       - Yes , both nn9989k

30. The job doesn't start, still pending. The jobid is increasing the whole time:
```bash
#!/bin/sbatch     <- this should be /bin/bash and not /bin/sbatch
#SBATCH --account=nn9989k
#SBATCH --mem=1G
#SBATCH --time=10:00
#SBATCH --reservation=nn9989k
#SBATCH --job-name=NINA
#SBATCH --error=output-%j-error.out
#SBATCH --output=output-%A-blah.out
echo -n "This script is running "

sleep 60
hostname
```
  - Was started with `./job.sh`
  - I would `scancel` it, fix the first line, and re-submit it with `sbatch job.sh`
  - The jobid changes too quickly to cancel with `scancel JOBID`
  - Worked with `scancel -u USER`
    - so problem solved? was the problem the first line?

31. Is it the same in the real scenario like the picture "HPC dinner", one job/user would be assigned with one core?
    - yes but we also have many jobs that ask for more than a core and wait until the number of cores they ask for become available
    - restaurant is still a good analogy. sometimes we reserve a table for 8 people and then the dinner party/group would be the "job"

32. General question: is it possible to copy all the output from the terminal to a file .txt using a command in linux?
     - and the output is the output of a command? You can redirect the output from a command to a file with ">". Example:
     - `ls -l` prints the directory content to your screen
     - `ls -l > directory-contents.txt` redirects the output to a file called "directory-contents.txt"
     - did this answer the question?
         - This is very useful and I did not remember exactly! But in the case I want to copy all the output (multiple command outputs, since the start of the session until the last line and output i got)? Is this possible? Thank you
           - The way I would do it is to select the output with my mouse, open a new file, and then paste it into the file (in my case with mouse wheel but it seems better with right click). Is there any better way? If I wanted to see all commands I typed, I would use `history` command but I understand here you don't want just the command but also the output.
               - Great I did not know if there was a more accessible way, but this works too! I see there is not also a way to select all lines like Ctrl+shift+up arrow
                  - I am sure there is a way to split all output to go both to the screen and also to some file but I don't think there is any more elegant way to recover "past" output since from terminal's perspective this is already gone and forgotten.
                      - Good, now all is clear! Thanks a lot

33. If I don't know how much time it needs for the job, and just give an estimated time period, e.g. for 48 hours, but it actually need 50 hours (longer than the estimated time), how could I handle this case?
    - The scheduler will stop your job.
    - see also question 16 on how and when we can help extending running jobs
    - Is there a way to avoid wasting 48 hours to get to know that the job needs more time?
       - what I do when it's possible: grow the job from small system size to real system size and that makes testing this easier and hopefully I can extrapolate to the real job. But it will still take some calibration and some trial and error.


## Q&A

34. How to run interactive jobs?
    - https://documentation.sigma2.no/jobs/interactive_jobs.html

35. Can you prolong/extent jobs yourself?
    - No, not after the job started
    - In special occations you can write the support team and we can extend it.
    - Also see question 16

36. Looking at interactive session. We start with salloc right? When do we use srun?
    - "depends" :-) but also see link under 34 where there are explicit examples
    - it seems we recommend `salloc` on all NRIS clusters but with slightly different options (e.g. on Saga one has to indicate the max memory)
        - Thank you!

37. How to estimate runtime?
    - Try smaller sample job (parts of problem, only one iteration) and then try to estimate
    - Ask for too much time in the start, maybe even maximum wall time allowed, e.g. 7 days. It will queue a bit longer but that's okay if you then know better what to ask for the next time
    - Always ask for a bit more time that you expect the job to run. So 8h or even 12h if you assume 6h. Once you are confident you acn go down a bit.

38. looking forward to hear more about how to run a program such as R or python, and particularly, how to execute scripts in it. Where do we find how we should actually use the HPC for the typical kind of jobs? I am not used to starting jobs on my own computer from the command line-so open a program, tell the program what to do, etc. I suppose this is a required skill for working with the cluster.
    - This topic is outside the scope of this course. Running R or python scripts on HPC is not much different from running them on your local machine, unless you have some special parallelization code
    - For the second part of your question : you shall refer to R or python documentation, both of them are made to be used from a command line, then these
    -  command-line commands can be executed on the cluster, just as we did this today.

39. How do I exit SAGA?
    - type `exit` and hit Enter

40. When to use interactive jobs?
    - Useful for debugging and establishing workflows and job scripts
    - But at some point you should have a job script to use instead of an interactive session. better for sharing, reproducibility and more robust
    - Useful if you want/need to test something without interference from other users and without interfering with other users' work since you can allocate interactive node exclusively only for you for a certain time
    - Useful if you need to debug a problem that crashes after few seconds and you don't want to queue for hours every time between each crash

41. How do I navigate from where I start when logged into Saga (`home/<username>`), to my project directory (nnXXXXk)?
    - You shall know the location of the project directory, then you use the unix command `cd <PATH TO THE DIRECTORY>` to get there, e.g. `cd /cluster/projects/nnXXXXk` where XXXX is a project number
      - Thanks, exactly what I was looking for.
    - One thing that helps me remembering where all the places are, is to add a symbolic link from my home folder to the other place.
        - A good idea indeed, you can also write a so called alias in your `.bashrc` file, a command which executes some particular `cd PATH` command. E.g. `alias my_proj='cd <PATH TO MY PROJ>'` Then you only type `my_proj` and you get where you want. After changes in `.bashrc` file you need to run `source .bashrc`.


## Feedback for today

- Please list one thing that you particularly liked
  - I liked that you did the instruction sessions as a duo. Overall a very good course! Thanks!
  - I liked the breakout rooms where you could ask questions
  - Many instructors to help out! Very nice to have all questions answered at once.
  - Its nice we can enter and leave any breakout room anytime so that we dont need to wait for our turn to ask a question
  - Enjoyed the live coding parts of the course.

- What we should improve/change for tomorrow or next edition
  - in the grey boxes it is not immediately clear to me what content is supposed to be part of the script, and which are commands to run outside of that
  - I was a bit confused what exactly the exercise was supposed to be. Maybe make that a little bit more clear next time :)
  - If you have some trouble with the commands/scripts it's a bit too fast and it may be hard to catch up. Maybe take more time to have the people that everything doesnt run smoothely for to follow
  - The goal of the exercises was a bit unclear to me.
  - It would be nice to repeat the session briefly after its done
