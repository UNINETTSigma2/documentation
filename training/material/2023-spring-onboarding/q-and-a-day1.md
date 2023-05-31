---

---

(training-2023-spring-onboarding-notes-day1)=
# Questions and answers from HPC on-boarding Apr 2023, Day 1 (Apr 19)

```{contents} Table of Contents
```
## Icebreaker Question
### Have you shared a computer with other, if yes did you face any issues?

- yes, one problem that came up is that I accidentally occupied the whole computer and nobody else could use it  
- Not a PC, but we have a local computing server which is very popular among our researchers. CPU is often saturated there due to processes that use more CPU than the user intended.
- yes, issue: folder structure changed by another user that messed up some scripts.
- Yes, ran out of storage (lab computers and servers)
- yes, no specefic issues
- Yes, no particular problems
- Yes, through remote desktops app. Only one user allowed at the time so we use a notepad to write who is using it.
- accidentally filled up the disk
- Yes, with my parents as a kid! A few issues came up I think
- Yes, issue with wanting to use it simultanously

## Sessions
https://documentation.sigma2.no/training/events/2023-04-hpc-on-boarding.html


:::info
Discussion (5 minutes):
Frequently, research problems that use computing can outgrow the capabilities of the desktop or laptop computer where they started:
:::


- Image processing of large images, with multiple processing steps, quickly exceeds the RAM of regular computers. This is especially true when additional dimensions are added, such as time lapse, or additional channels/colors.
- Yes, working with images as input. Hyperparameter search / tuning networks. Could only train on top layers.
- bioinformatics analyses
    - Bioinformatic analysis, from large sample set of metagenome sequences to abundance table stage.
- Image processing of radiographs, just starting
- Molecular dynamics simulations
- Bioinformatic analyses /de novo assembly of bacterial isolates and phylogenetic analyses
- Climate simulation / weather forcast
- Gene annotation of NGS data, especially PCR error correction taking up a lot of RAM
- Large datasets in psychology / neuroscience
- Simulations with lots of parameters
- Bayesian models using MCMC
- Image processing of large field of views


## Survey - Found my ssh
Answer with o 
- Yes: 0000000000o000000
- No: 
- It is working after I log into my FRAM account, doesnt work before the login (then I get an error)
    - Hm, but still you manage to use ssh to log into FRAM? That is good at least! Yes I use ssh to log into FRAM. But befor shen i typed "ssh - V" i got  "ssh: Could not resolve hostname -: nodename nor servname provided, or not known"
    - there should not be a space between the dash and the V `ssh -V` is the correct command
    - yes, it is because you have a space between the dash `-` and the `V`, :) 
- aa thanks! :+1:


## Have you logged in to Fram(or any other HPC system)

Yes: 0000000000000
No: 0
 - Permission denied, please try again. (username: USER_NAME@fram.sigma2.no)
 - Is this the first time you tried to login? 
     - yes
 - Are you sure you have/are typing the correct password? 
     - yes, also tried to reset password. Its the same user as to metacenter?
 - Did you get a new account for this course, or did you already have an account?
     - new account, got email today saying: The access for user ****** to project NN9987K expires on April 20, 2023.
 - Right, so new
 - Ok, will have a look at this and get back to you
     - It worked when I dont copy paste the password....  

-----------------------

## Your questions

1. What are the advantages of faster I/O capacity, other than reading and writing data more quickly (which might happen only at the beginning or end of a script, so may not be a huge deal)?
    - The I/O capacity is in many cases shared with several users.
    - It is also not always the case that this is only done at start and end of a script. Quite a few workloads needs to access data during execution, and a few, quite a lot.
    

2. Not a question, but about Maikens story about using HPC without knowing to do so. I would hope the system would be more like that. Not necesarry to know Unix and all that. Hide some of the complexity from users and let them concentrate on data and algorithms.
    - Yes, more and more the access is getting like that with web-interfaces and ways to construct your workflows and jobs using drop-down menus etc. Examples are Galaxy portal, Ondemand portal and many more
    - Thank you, will we learn about those in this course?
    - No, unfortunately, but this will definately come at some point, it is a bit new at the moment, but later tutorials will be expanded with this type of things
    - If you are interested in Galaxy, there is a self-paced community course (or what to call it) in May: https://gallantries.github.io/video-library/events/smorgasbord3/

3. at the end of yesterday lecture, Espen mentioned two software to connect to hpc i think, could you please repeate the name again. Thanks
    - were these graphical tools? (sorry I wasn't here yesterday ...)
       - yes they were related to GUI, but both also offers terminal capabilities. And support standard ssh.
       - X2Go: https://documentation.sigma2.no/getting_started/remote-desktop.html?highlight=x2go
       - Visual Studio Code: https://documentation.sigma2.no/code_development/guides/vs_code/connect_to_server.html?highlight=visual%20studio%20code
            - perfect so they are similar to remote desktop?
               - Yes they are remote desktop also. X2Go is our remote desktop solution that we support. Yes
               - VS code is a code editor, and when connecting to HPC, you can access all your files as if they were on your computer
                   - could you please demonstrate X2GO once ? i have tried myself but i think it is not working or at least i did not recognize how it is working.

4.  Is it okay to be in login-1.FRAM? 
    -  Yes it is perfectly fine. fram.sigma2.no is a generic portal that directs you to different login entries depending on workload. We currently have more than one, since there are many users. login-1.fram is the name of 1 of the login nodes, login-2.fram is the name of another. You may also log in using user@login-1.fram.sigma2.no if you need to go to a specific log in server (or 2, 3 etc).

5. where can the server (hash) key be verified (displayed on first login)?
   - https://documentation.sigma2.no/getting_started/fingerprints.html

6. On [this image](https://training.pages.sigma2.no/tutorials/hpc-intro/_images/hpc.svg), is it possible to identify a single node? Is it one unit on a rack? Is it a whole rack?
   - one unit on a rack. what I am unsure about is whether one of the "pizza boxes" (the 2-3 cm high metal things) host 1 node or more than 1 node. 
       - What you refer to is probably the single hight, this contains two compute servers, typically also called nodes. Due to technical reasons we store two and two on each level. Each node for our machines mostly contains two cpu´s with a number of core units - ranging from 16/cpu on Fram to 64/cpu on Betzy.
    - should I go down and take a picture of a node on fram?
        - that would be nice. we could also put it into the documentation. :+1:
        - ØT: On my way (not available for breakout rooms)
        - Picture of a rack in fram
        - Each one contains 6 chassis
        - ![](chassis_fram.png)
        - a chassis
        - Each containing 6 blades
        - ![](blade_fram.png)
        - a blade
        - Each containing 2 nodes
        - ![](insidenode_fram.png)
    - thanks for posting these!
    - This was very cool to see!


7. Do you pay for using the "login node" or only the "working node"?
    - The cpu-time is calculated from the actual analysis job you submit via the batch system. So that is the time that will be accounted towards your project quota. 
    - in short: for the compute node, not for the CPU time spent on login node. but this does not mean we should compute on login node :-)

8. Are all servers in Fram physically located in the same room? What about Fox, Saga, Betzy?
    - the compute resources: yes
    - this is also true for the other clusters. one reason why a compute cluster is typically in one room and not spread across several cities is that the high-speed network ("interconnect") which connects the compute nodes, puts a limitation on the length of cables.

9. I dont understand the writeout I get after the sinfo command in the terminal. Could you explain?
    - it gives us an overview over the available queues (or partitions) ("normal", "optimist", "bigmem", ...) and a mapping between the queues and node names ("c3-1", "c2-10", ...) and their status ("idle", "drain", "alloc", ...)
       - maybe the actual information it gives is more useful to those maintaining the system than to those using the system
       - The information will be different depending on what system you are on, so SAGA, FRAM, Betzy will have somewhat different queues and specifications  
       - You might find this useful: https://documentation.sigma2.no/jobs/overview.html

10. The queue manager, you called it "Slerm"?
    - [Slurm](https://slurm.schedmd.com/documentation.html)
    - thanks! on which page is the typo?

11. Are "schedulers" and "queue managers" the same thing?
    - yes, same concept

12. My example_job.sh doesn't want to execute, it says permission denied. I made it executable. Any tips? :)
    - if you like you can post the full path to the example script and I can look as staff (I will then also redact the path away again)
    - Remember to run it like e.g. `./example-job.sh` the dot and / if you are in the same folder
    - Run pwd to get path
      - path: (redacted path)
       - thanks! looking ...
       - solution: `chmod +x example_job.sh` (to make it executable)
         - but I did that (several times lol)
    - how did you try to run it, and what doesls show you? See how to do this in the day 0 course: https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/scripting.html

   - Can I get the output of `ls -la`
     - currently the file is not executable
       - now it is executable 
     - So it is the chmod +x that gives you the permission denied?
     - I called this: chmod -x example_job.sh
        - (See also below) `-x` removes the executable "flag" (information) 
     - then: ./example_job.sh
     - it's the script execution that gives me the permission denied error
     - I made it executable now. Try to run it again `./example_job.sh`
     - thanks it works now, but it doesn't really explain why my chmod didn't work?
        - No, it does not... I have to admit, I have not seen this problem before... You should have the right to make something executable in your own home folder..
        - I have created a file named `test_permission.sh` in your home folder, could you try to run `chmod +x test_permission.sh`?
        - I'll try again with another script. Yeah, the chmod seems not to work, though it gives no error or warning. Okay got it now, the problem was that I used "-x" for "+x" Sorry!
          - yes, + and - will do what you think. `chmod -x file` would remove eXecute permission from a file, + adds it
          - Exactly. I think I need some coffee.
             - somebody said/wrote: there is good pedagogy in every mistake. we all learned something new.

13. I only see nn9987k
    - hopefully you can use this quota for the exercise. in other words: nothing to worry about

14. What's the difference between projects ending in "k" and those ending in "o" (just curious)?
    - my understanding is that all compute projects normally end in "k" but optimist quotas end with "o" (these can only be used on "optimist" queue?)

15. Adding details to a command seems to come with a - (eg -l or -n). What does -- indicate?
    - many commands use one dash for short options and two dashes for long options (long meaning: option is written out). example: `-A` or `--account` do the same thing
    - See Day 0 information: for instance https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html and also the cheat-sheet: https://training.pages.sigma2.no/tutorials/unix-for-hpc/cheat-sheet.html
        - Neither of these mention the -- (other than having to override the missing certificate), but the other explanations in the responses here answer my question.
    - Ok, does it not say explisitly, at least it mentions options and shows the options using the `-`  - but good that you understand it now! :) 
    - It is typical syntax of UNIX (and probably also Windows) commands, that you use `-` or `--` to give options to the command

16. Interesting for customation of jobs: 
    - https://open.pages.sigma2.no/job-script-generator/
        - Is it possible to add the time units in the documentation somewhere, as in hh:mm:ss ?

17. why does all the example files end with .sh? would it end with .py if it was a python file?
    - the ending here only helps us humans to anticipate what is inside the file. `.sh` communicates to me that there will probably be a shell script inside. but for the cluster it does not matter. I could call it `example.run` or `example.batch` or `example` or `example.job` and it would still work. If it was a Python file, it would be good to call it `something.py`, again to communicate expectations.
      - okay, so it does not specify "what language" to read the script in?
          - correct. but look at the first line inside the file that starts with `#!`: this one tells the shell which interpreter to use to interpret what follows. if that was a Python script, it would contain `#!/usr/bin/env python`
        - Thanks!
          - a bit more explanation (hopefully more helping than confusing): I can run a bash script like this: `bash somescript` or like this: `./somescript` - in the latter case the shell does not know yet what language to use and then it will look at the so-called "shebang" line (the line starting with `#!`). in other words, if I run a Python script like this: `python somescript.py`, then it will NOT use the "shebang" line.
          - the take-away for Slurm job scripts: we can use any language which uses "#" for code comments

    - See also yesterdays tutorial which explains this: https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/scripting.html

18. I am unsure how to estimate 1)the time a job needs and 2)how much CPU hours it will cost. Do you have some resources for that?
    - this is a great question. it typically requires some calibration. what I personally like to use is to start from a small example and grow it to the real example. often I can extrapolate from the smaller example towards the real example. but it often requires trial and error. for the first run I am often more generous and then after the first run I know how much time the other runs will use - was this helpful?
        - Yes, thanks! I find myself being "scared" to try, what happens if we run out of resources on the project account? Will the job stop or will it continue running and then we get a huge bill?
        - You will only be billed for the actual amount of cpu-hours spent. So the "only" bad thing about specifying too long time is that 1) it may take longer you your job to start and 2) it prevents SLURM from doing a good job allocating other jobs since maybe another that that was short could have fitted onto that node, if the time you specified was shorter (and correct).
    
19. Can results from a terminated project (which exceeded its time limit) still be accessed, for example at a shared storage area?
    - We have a shared storage system for all nodes. So, if you save something to a file, it will be available even if the job crashes. But you will have to actually save it to a file. Results in memory will be gone.
    - It is a rather tricky and actually quite advanced question: Depends a bit on your job-setup. Have a look at the runscript example on our main documentation, for instance the gaussian runscript example under application support to see a solution to keep the files even if the job dies prematurely.
        - Link to main documentation: https://documentation.sigma2.no/
            - This one? https://documentation.sigma2.no/software/application_guides/gaussian/gaussian_job_example.html
    - Will the shared storage system be covered in this course, or should I refer to the main documentation?

20. Is a possibility to use cpus from two or more projects for our job? And if yes how we command this in SBATCH?
    - to my knowledge: no. a job will use one account
    - Depends a bit what you mean by project. But as a compute project with allocated compute resources/CPUhrs - the answer above holds.
    -Yes, I mean an account.

21. How do you know what %j means? What would the source to look for all the combinations such as %a or %A.
    - You can get that details from SLURM sbatch manual
        - https://slurm.schedmd.com/sbatch.html 

    - Is it possible to add a timestamp? I cannot find that in the doc.
        - That you need to use UNIX tricks to fabricate a file name
            - ```filename=$(date +"%m-%d-%H-%M")"-somename.txt"```
            - The above variable *filename* would then contain the value 04-19-11-17-somename.txt

22. There is a part on running interactive jobs in the end of Sabrys lesson. Is this to use with for example Jupyter for debugging? I am still in an early phase and my Python programs only give me a lot of errors, so I cannot submit them as scripts yet.
    - Thank you for pointing it out. Will show this after the exercises    


23. If one wanted to submit 10 or 20 calculations which differ in a file name but not in the ending (file suffix), how can one do that? assuming that the calculations have same resource requirements but only differ in some calculation parameters
    - one option is Slurm job arrays where you can write one job script but inside the job script, Slurm will define an index number for each job which one can use but the index is numeric. this way one can inside the job refer to files. [Slurm Array jobs](https://slurm.schedmd.com/job_array.html)
    - alternative: one (bash) script (but does not have to be bash) which iterates over the files and for each submits a job script. (I will post here an actual example in a moment ... just thinking about how to not make this confusing):
        - "outside" script:
      
        ```bash
        #!/usr/bin env bash
        
        # here assuming that all files that I want to submit end with "job"
        for file in *.job; do
            sbatch $file
        done
        ```
        - however, disadvantage of this approach is that I now need to write many similar job scripts. But there is a better way to avoid that. In this example one "outside" script generates 3 scripts, each 1 minute long, and submits them all at the same time.

        ```bash
			#!/usr/bin/env bash
        	# instead of iterating over greetings we could iterate over
			# some parameters
			for greeting in "hello" "hi" "ciao"; do
		    # Generate the SLURM script to be submitted
   			 # "EOF" means "end of file"
    		cat > submission_script.sh << EOF
			#!/usr/bin/env bash

			#SBATCH --account=nn9987k
			#SBATCH --job-name=example-${greeting}
			#SBATCH --output=${greeting}.out
			#SBATCH --ntasks=1
			#SBATCH --time=00:01:00

			echo ${greeting}

			EOF

    		# Submit the generated script
    		sbatch submission_script.sh

			done
			```

- There is also the option to write a job script generator which iterates on many different file names. This is quit widely in computational chemistry. However, this requires a somewhat more advenced insight in linux power tools/scripting. We may provide example, but it would be on your own risk type offer. 

- using piping, you can generate the list of input parameters you need and create a file that has that list. As the array job wil get its input index, you can use this index to look up in the file what filename that exact job is supposed to use.
    - SLURM_ARRAY_TASK_ID is the index in the job array.
    - Found a better tutorial: https://blog.ronin.cloud/slurm-job-arrays/
 

24. Could you give us an example for `--ntasks=<ntasks>`? I mean, a job that would benefit of multiple processors.
    - example in which we ask Slurm for 20 cores. `srun` passes the information (how many cores are available) to "mycode" which in this example might be an MPI-parallelized program (MPI is one way out of many to parallelize):
    
```
#!/usr/bin/env bash

#SBATCH --account=nn1234k
#SBATCH --job-name=testing
#SBATCH --time=00-00:02:00
#SBATCH --ntasks=20

srun ./mycode
```
    
   - the bigger picture and more important answer is that just by asking Slurm for more processors (with `--ntasks` or similar options) will not make sure that also the code runs in parallel UNLESS the code is ready to also use these resources. this often requires either looking into the code or checking the documentation of the code.


25. I'll have to leave before the last session today: When will a recording be available?
    - Yes the recordings will in Youtube, but it will take a week or two 


26. I wonder about the #SBATCH inside the script vs the sbatch as a command line argument. When do we use what and why? 
    - both have the same effect. on the command line it can be useful for quick testing. but after the quick testing I always prefer to put this rather into the script because then when I come back to my job script 6 months later, I know what I actually did. By then, the command line and my own human memory is long gone. For reproducibility, scripts are better.
    - `sbatch` on the command line will actually execute the sbatch command. `#SBATCH --someoption` in the job script file just specifies that the sbatch/srun command is supposed to be run with the `--someoption` option. It is a way of seperating what 'bash' is supposed to handle and what 'slurm' is supposed to handle 


27. How is the time specified: hours/minutes/seconds in the sbatch command?
     - example: `#SBATH --time=02-04:00:00` submits for 2 days and 4 hours (DD-HH:MM:SS)
    - Thank you. Where can this kind of information/documentaion be found? 
        - https://documentation.sigma2.no/jobs/job_scripts.html
        - or: `sbatch --help`

28. How can I check how much memory, CPU etc my job needs? While running or after it is finished?
    - `seff 12345` (where 12345 was the job number), after the job ends
    - While the job is running you could login to the node(s) that runs the job and use the command ```top``` or similar UNIX comands
    - see also our tutorials on this very good and important question:
      - https://documentation.sigma2.no/jobs/choosing-memory-settings.html
      - https://documentation.sigma2.no/jobs/choosing-number-of-cores.html

29. We assume that VS Code is preinstalled and set up if people plan to use it during the transfer of files session. Please follow links above. 

30. Why are all my test jobs stuck at "PD" state?
     - "pending" ... can you share job number(s) and then we can look for reason?
         - Sure, job IDs are (redacted) (assuming that's the same thing as "job number")
            - thanks! looking ...
            - jobs look good and my own test job is also pending. it seems jobs are waiting for resources. maybe we were recommended to also use a reservation in the job scripts? (but I wasn't here yesterday ...)
      - Did you use ```–partition=course –reservation=NRIS_HPC_onboarding```
          - No, forgot that. Thanks!
          - Then cancel the job and submit it again with VIP pass above
          - Done, that does the trick. Good to have learned about different job states, anyway. :)

31. I missed a comment early on. Is the reason for using `rsync` instead of `scp` is because the former is much faster or is the latter not supported on Sigma2?
    - It is supported, just faster and more safe to use - rsync typically picks up where it left if you lose connection, and it also just updates the files that are actually changed, while scp copied files no matter if they are changed or not (changed with respect to the state on the remote site)

32. when I have used rsync previously, rsync says "not all files were transferred" (or something like that). This only happens to a few of the files i transfer. Is there something that may be cause by the rsync? Does it not work with all file types? Would it help to zip beforehand?
     - zip beforehand should not be needed. I am wondering whether the information could mean that rsync did not transfer all files since it did not need to? It does not transfer files which are already there. It looks at file size and time stamps and can also consider checksums. But I am not sure about my answer. I haven't seen the error you describe, yet.
         - based on what Øystein just said: I am transferring a folder, do I need to specify -r when transferring a folder to get all files?
             - the `a` is used to tell rsync that you should consider files in folders and subfolders - in `scp` () it would be `r`
                 - ah yes
             - according to my limited knowledge, no. example for transfering files and folders: https://documentation.sigma2.no/files_storage/file_transfer.html#transferring-files-between-your-computer-and-a-compute-cluster-or-storage-resource

33. Som you dont need to be logged into FRAM to move something to FRAM? 
    - that's right. you still need to authenticate, though. If you use SSH keys, then the authentication can happen "under the hood" and it can look like that you can transfer files without providing a password. 

34. my username for this course is ...@login-3 and not ..@sigma2.no, should i specify the whole "email" or just what comes before the @?
            - hmmm... I am confused by this. Usernames should not contain any "@". once you log into a cluster, your username shows up as the part before the "@". you can also try command `whoami` to check what your username is
            - Thank you. I'm just asking because I think Espen wrote his username with the @
               - ah, I was distracted and did not follow screenshare
 
34. No rsync in Windows GitBash?
    - I am unsure but rsync should be available in Windows Subsystem Linux. 
    - where do I find this subsystem? GitBash says its using Mingw64 bash: rsync: command not found
        - https://www.ubackup.com/windows-10/rsync-windows-10-1021.html


35. is there a setup tutorial someone could link to? 
    - Yes! https://documentation.sigma2.no/code_development/guides/vs_code/connect_to_server.html

### Survey - Interested in seeing moving of files GUI type with VS Code:
Answer with o 
- Yes: ooooooo

36. I had to leave early and will not be able to attend tomorrow. Please tell me where I can find the recordings. Thank you for such an informative course - Gerald
37. 

(Assume that "not yes means no")

Will be displayed during exercise session and/or after todays lession if pressed on time. 

```{note}
The example shows `mem=1G`, but this will not work on the current course allocation on FRAM. However, it is useful for you to then see what error slurm gives you. 
```

### Feedback

 * Things that went well
    -
    -


 * Things to improve
   -
   -

