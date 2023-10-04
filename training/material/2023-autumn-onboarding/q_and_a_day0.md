---
orphan: true
---

(training-2023-autumn-onboarding-notes-day0)=
# Questions and answers from HPC on-boarding Oct 2023, Day 0 (Oct 3)

```{contents} Table of Contents
```
## Your wishlist

- I would like to understand better how to choose the appropriate parameters to run a job to allow a good gestion of the resources. So, fare I am using the same parameters for every job witch I guess is not the best. 
```
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8"
```
- 
    - We will talk about the parameters. In detail, you can follow our documentation here.
    1. [How to choose right amount of memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html#choosing-memory-settings)
    2. [How to choose the number of cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html#choosing-number-of-cores)
    3. Video recording on this can find [here](https://www.youtube.com/watch?v=m7u52vIUwwc&list=PLoR6m-sar9AgoSnXdbUkO9FBiCUTft1GK&index=5)
    4. Also note: The next level NRIS course typically go through this. We are having some restructuring of courses, but it used to be called HPC Best Practices. 
    
- I would be interested in running a program for which I have a licence on the cluster. It is called YASARA and is used for protein ligand docking and basic MD. Could you please let me know if this would be possible?

   - If the licence allows it, this will be possible. For more assistance, please contact our support via support@nris.no

- Some introduction to running program like Star CCM+ or ANSYS or OpenFOAM etc

   - We won't be go into detail about this softwares, but you will get general overview of how to acces these software. Further, you can follow our [documentation](https://documentation.sigma2.no/software/application_guides/openfoam.html#running-openfoam)

- I am looking for help regarding use of HPC resources of TSD and sigma2 for using deep learning application with GPU.
   - Kindly follow the [GPU documentation](https://documentation.sigma2.no/code_development/guides/tensorflow_gpu.html#index-0)
   
- I will be using WRF in Linux for my master thesis but using the Myrrha HPC. I just want to be very comfortable and confident with using Linux. Also, it will be very useful for my career pursuits.

  - You are welcome!

- I specific wish is to learn linux command line and to find and use different softwares in Saga to perform gene sequencing analysis.

  - You are welcome!


## Icebreaker

- How do you ask a computer to do something for you?
  -  I use the mouse to point and click to open the programs I want to use
  - With scripts and programs because they will remember even when I forget later
  - guiding with the mouse, clicking to open any programs or files
  - Tell the computer what to do through the terminal

## Polls for introductory lesson (vote by placing an "o")

### Locating the terminal

*  I found my terminal : ooooooooooooo
*  I can not find my terminal :
*  I am lost, what is a terminal? :


### Locating ssh (type `ssh -V` on your terminal)

* I have ssh : oooooooooooooooooo0oo0ooo0
* I do not have ssh :
* I am lost, what do you mean by ssh :
* I know what SSH is but I am not sure what you mean by "locating ssh" : oo 
* All clarified now? o

### Did you manage to login in to saga
* yes:oooooooooøooooooooo
* no : 
* no and I need help :
* no I don't have an account on saga: 

## The terminal and remote login

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/intro.html

1. I could not get terminal using ctrl+Alt+T windows
   - which system are you on? Windows/Linux/Mac?
   - Did you find it in another way? In the program search?
     - Yes using the search - good

2. Other than a basic Terminal, which other tools/software (free) would you recommend (I heard about MobaXterm, Termius, etc. but have not tried them)?
   - for windows users who might be completely new to terminal, I would recommend having a look at MobaXterm or VS Code  with remote SSH plugin
     - https://documentation.sigma2.no/code_development/guides/vs_code/connect_to_server.html 
   - terminus looks interesting. I did not know about it.
   - for mac, I would suggest 'iTerm' 
   - for windows, the Windows Terminal (https://github.com/microsoft/terminal) is also ok
 
3. How do you access saga?
    - If you have a user account, you can use the command on your terminal ` ssh yourusername@saga.sigma2.no` . For the first time, you will be prompted to affirm the fingerprint. You can check it [here](https://documentation.sigma2.no/getting_started/ssh.html#first-time-login) and then you will prompted to your password. The instructors will show you this soon. 

4. Where to find the fingerprints?
    - https://documentation.sigma2.no/getting_started/ssh.html#first-time-login
    - [Fingerprint of our systems](https://documentation.sigma2.no/getting_started/fingerprints.html)

5. I'm getting permision denied; I don't know why?
    - Are you a new user? Can you contact the host(dhanya) on zoom with username?
    - Dhanya is checking
    - Resolved

## Moving around and looking at things

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html

6. Should we all type with you or only watch?
    - only watch for the moment
    - but if you prefer to type-along, you can
    - there will be exercise sessions where we can try it

7. Do we have any cheatsheet with all commands that we use?
    - https://training.pages.sigma2.no/tutorials/unix-for-hpc/cheat-sheet.html

8. Does the fingerprint have to be identical to the fingerprint in the link? Mine starts with SHA256: but the rest is not the same as in the link. 
    - It should be.
  - How do i fix it?
    - mReport back to support@nris.no. Do you have ed25519 fingerprint also?
    For this case, it is most likely fine. 
mine is: SHA256:qirKlTjO9QSXuCAiuDQeDPqq+jorMFarCW+0qhpaAEA. They are identical, see https://documentation.sigma2.no/getting_started/fingerprints.html#ssh-fingerprints. No worries!!

9. my temporary password will expire today, passwd command to change password is throwing error. Help needed
```
passwd 
Changing password for user myusername.
Current Password: 
New password: 
Retype new pange failed. Server message: Insufficient access rights

passwd: Authentication token manipulation error
```

- Try this: https://www.metacenter.no/user/reset/
    Thank you!
   - Is it resolved? -- yes, I got another temporary password which will expire on 10.10. 
   - You can change it  https://www.metacenter.no/user/reset/.
   - Contact the host on zoom private chat, if you still have trouble.
   - Now I am able to change my own password. I can access the server with my new changed password.
   - Great. Thank you for responding :thumbsup: 

## Exercise

- https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html#exercise

### Status
* I am done: ooooo
* I need more time:
* Not trying this one: o 


10. I cannot find any .output files in any subfoldeer?
    - ~~Indeed. I am not sure the exerice and the solution matches demo-moving-around. We will ask instructors to double-check.~~
        - Observation is correct. The files download are different for exercises 
            - aha I extracted the wrong thing. the example on top of the page is different than the exercise tarball. `demo-moving-around` is different than `exercise-moving-around`.
            - aaaaah, I made the same mistake, thank you!
                - Yes, we will fix this confusion in the future
    - you will find the .output files if you list the content `ls ~/exercise-moving-around/results`


## Finding things

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/finding-things.html

11. Didn't quite catch what is safer about `*.csv` vs *.csv, and what safe means.
     - good question. Let's see if we can come up with an example that demonstrates the difference.
     - Example:
       - go into a directory which contains a couple of .csv files, for instance `a.csv` and `b.csv`
       - try: `find -name *.csv` - this will fail because the terminal "expands" this to `find -name a.csv b.csv` before the `find` command is run and in this case find does not know what we mean.
       - then try: `find -name "*.csv"` - this works. here the terminal does not expand the wildcard but it is `find` which deals with it.
    - did this clarify the difference? 
    - "safer": it does what you expect instead of being unsure what might happen or doing something unexpected or giving an error

12. What does `grep` stands for? 
    - `grep` stands  for Global Regular Expression Print :+1: 

## Exercise :
https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/finding-things.html#exercise)

### Status:
* I am done: oooooooo
* I need more time:

13. After restarting my computer i get Command ‘ssh’ not found
    - What OS (Operating System)? 
    - If it is windows, Open your settings. Go to Apps > Apps and Features > Manage Optional Features. Check the list to see if “OpenSSH Client” is enabled. If not enable it.

## How to create and modify files

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/writing-files.html

## How to compose commands with pipes

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/pipes.html

14.  how do i input a pipe in windows?
    - if a Norwegian keyboard, press the key to the left of number 1 
    - yes i got it
    
15.  How so you make the pipe sign on mac?
     
    - On mac you can type a pipe symbol by typing (english keyboard) `shift + \`
    -  For me it is option+7 (Norwegian keyboard)

16. Should all commands be on the same line when using pipe ?
    - Yes, if you press enter the that would mean executing it - in a script you can break the line by using a line-break character though `\` but sometimes it does not work completely as expected (sensitive to spaces etc), so you can experiment with that

17. Will the histrory be there when I login next time ?
    - yes. it is preserved across log-out/log-in. however, it has a max length which can be re-configured. I think my history is configured to keep the last 1000 commands I typed.

18. How do I use pipes with my own commands, instead of UNIX commands? can I use python scripts ?
    - yes, for example: `python myexample.py | grep "something" | sort`

19. When you used the magic xargs, how do I prevent overiting when I want to direct the final output to a file ? e.g. collect last 10 lines matching something to one file 
    - You could use the double angle brackets `>>` if you want to write to the same file. When we use that all new information will be appended at the end of the file
    - If you used a single angle bracket `>` then the new informaiton will ovewrite the existing content. So if you do this only the last thing you wrote will remain in the file, everything elese would have dissapared, Please note UNIX operations like piping contents to a file has no `undo`
     

## Scripting 
- https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/pipes.html

## Exercise
- https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/scripting.html#exercise

## Status
- I am done: ooooo
- I need more time and I will do  later today:

20. I am not able make the new script
    - You can try `nano filename.sh` to edit your file, then add the commands,  here you can follow the solution . 
    - Thanks, i managed it with your suggestion
    - You are welcome!

## Feedback of the day

- One good thing about today
   - very organized and easy to follow 
   - Very nice online course because I could learn several commands that I did not know. Thank you very much!
   - Technical setup is great! Excellent tutorials make it easy to follow the tutorials and refer back when unsure.
   - the exercises are great and I like the tutorials
   - I thought the tempo and level were great. Not too fast or slow. 
   - Thank you for the cool new things to learn (but I don't know how would it be for totally new linux users :) )
    - very informative! thank you <3 
    - Very useful intro for new unix users. I think it will be very useful for tomorrow and Thursday! Thanks!
- One thing we should improve
    - for me as a beginner it was sometimes a bit too fast
    - would be nice to spend a bit more time on each topic
    - Documentations (i.e.,online text) are still being updated, so it would be nice if you could indicate the version in the text.



