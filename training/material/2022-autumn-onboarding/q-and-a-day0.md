---
orphan: true
---

(training-2022-autumn-onboarding-notes-day0)=
# Questions and answers from HPC on-boarding Oct 2022, Day 0 (Oct 18)

```{contents} Table of Contents
```


## Your wish list

1. How to access and use HPC for bioinformatics work (e.g. RNA-seq analysis using software such as Trinity)
    - we shall plan a HPC Q&A session for bioinformatics
2. Would like to follow the hands-on exercises
    - great, we will do that during the course
3. when and how to use multithreading jobs
    - we will discuss this in the follow up course [Best Practices 1-3 Nov](https://documentation.sigma2.no/training/events/2022-11-best-practices-on-NRIS-clusters.html)
4. Debugging and profiling HPC programs
   - see answer to question 3
5. Using Visual Code Studio with the national HPC systems
6. Fast storage available in the national HPC systems
7. RDMA support in the national HPC systems
8. GPU clusters available in the national HPC systems
   - There will be separate course for GPUs at NRIS. See the [documentation](https://documentation.sigma2.no/getting_started/tutorials/gpu.html)
10. A couple of words about slurm scripts would be nice.
    - great! we will do on day1
11. I'm working with IDUN, so highlighting some differences and similarities between IDUN and Saga/Fram/betsy would be nice
    - let's see our NTNU colleagues can say something
12. I would like to learn about Slurm to enhance the speed of my job. Also, it would be great if you could provide more specific courses regarding bioinformatics, such as Docker, Nextflow, etc.
    - basic slurm parameters will be discussed during the course. And Part of this will include this in the follow up course [Best Practices 1-3 Nov](https://documentation.sigma2.no/training/events/2022-11-best-practices-on-NRIS-clusters.html)
12. How to apply for computing time. And succeed. :-)
    - [documentation](https://documentation.sigma2.no/getting_started/applying_resources.html)
    - but also really good idea is to reach out to support near you and get feedback on your application before you apply. especially first-time applicants often don't know the answers to questions they get asked.
14. Ways to estimate resource usage before submitting jobs, identify when/whether RAM vs processors are limiting the speed of a job completion
    - Part of this will include this in the follow up course [Best Practices 1-3 Nov](https://documentation.sigma2.no/training/events/2022-11-best-practices-on-NRIS-clusters.html)
15. I would like to learn about how to apply for resources and how to access and use the HPC environment.
    - We will plan something for next semester. For the time being see the [documentation](https://documentation.sigma2.no/getting_started/applying_resources.html)


## Icebreaker

- What are the methods you know that we can use to interact with a computer/phone/smart device (i.e. how can we tell a computer to perform some task for us)?
    - I use voice to open maps
    - Terminal
    - Keyboard
    - Hey Siri, Hey Alexa...
    - Ansible
    - putty terminal
    - script

- What do you know now about HPC/Linux/Unix that you wish somebody told/taught you earlier?
  - write a script for anything non-trivial so that I can find it again
  - Job scheduling

- What aspect of Unix/terminals do you hope to get clarified today?
  - how to compose commands


## Questions

1. Clicking on the slides for the presentation doesn't load the slides
    - The course material can be found [here](https://training.pages.sigma2.no/tutorials/unix-for-hpc/index.html) and [here](https://training.pages.sigma2.no/tutorials/hpc-intro/index.html). Also, please see the [official page for this course](https://documentation.sigma2.no/training/events/2022-10-hpc-on-boarding.html#course-material).
    - where is the link to slides which does not load?


### Terminal and remote login

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/intro.html

2. I have just applied for user account. I haven't received confirmation yet
   - in this case you can still get most out of the course by watching the instructors type. I don't know how long it will take for you to get the account
   - could you send an email/Direct message to host? I shall request to speed up the procedure.

3. is bash and shell scripting the same?
   - for our purposes it will be as on the clusters the default shell is bash. however, bash is not the only one. there is also zsh, fish, csh, and some other older shells around. they are all similar and have their own pros and cons. bash is the most popular shell/terminal language.


### Poll: terminal

You managed to open up a terminal/shell on your computer (add an "o"):
- yes: oooooooooooooooooooooo
- no:

4. What is the relationship between putty and the terminal?
   - putty is a program that gives you a terminal interface on Windows. a nice alternative on Windows is either [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) or https://mobaxterm.mobatek.net/
   - but putty is basically a terminal for Windows but it can do more than that

5. What is the difference between windows power shell and command prompt on windows?
   - on Windows there are many command line interfaces with their own languages: power shell, command prompt, WSL (see above), putty and others, git bash. in this course we are *not* using neither the power shell, nor the command prompt but [Git for Windows](https://gitforwindows.org/)

6. can i use git bash in windows?
    - yes, [Git for Windows](https://gitforwindows.org/)


### Poll: ssh

7. You got a successful attempt at `ssh -V`
   - yes: ooooooooooooooooooooo
   - no:

8. What does the V stand for?
   - Version
   - ahhh
       - Most programs follow this standard -V
       - many/most tools use `-v` or `--version` but ssh is a bit different

9. ssh fingerprint can be found [here](https://documentation.sigma2.no/getting_started/create_ssh_keys.html#sha256-fingerprint) (for verifying the connection)

10. Is there a difference between connecting to the cluster with putty and some other terminal?
    - not really. once you are connected, it will behave the same and look the same. so if you managed to connect, it is fine and no reason to change it now
    - different programs can differ in the way they allow to copy files or having multiple terminals open at the same time

11. can we use vnc viewer for saga for graphical login?
    - in principle yes but I admit that I don't know the latest status of that interface. what I know is that my colleagues at NRIS are working on a new tool to make that even easier, smoother, and more robust but I am not directly involved. but I will reach out and ask.


### Poll: login into Saga

Can you log into Saga and verify with `hostname`?
- yes:ooooooooooooo
- no:
- I don't have a user account on saga: o

12. Regarding the suggesting to apply now and get access tomorrow: the registration is closed is it not?
    - Registration to the course is closed, but you can still apply for an account for this course. We will pick that up and process it as quick as possible. Follow the instructions that you got in the course email.

13. I just got my account :)
     - yay. welcome! 😃


### Moving around and looking at things

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html

14. We are all now expected to type with you?
    - you can but it is not a problem if you get stuck or if you prefer to only watch as during the exercise you can try it with a helper around

15. How do you scroll through the manual pages?
    - either page down and page up (use "q" to quit)
    - or: CTRL+F (forward) and CTRL+B (back)
    - or: up and down arrows
      - aha. nice. I am not sure I ever tried

16. How did you stop the command you typed
     - I didn't watch closely but I abort commands with CTRL+C so in the linux world CTRL+C is not for copying but for aborting ("C" for cancel)

17. If the folder doesn't exists, will it create one?
     - please give more details about "it" (I was distracted and not sure what exactly you mean)
     - Copying into a non-existing folder does not work
     -
         - aha thanks. no if the target folder does not exist, it will complain or create a new file with the content instead
              - depends. if I do: `cp somefile somefolder/newfile` and `somefolder` does not exist, it will just complain and not create anything but indeed `cp somefile somefolder` creates a file `somefolder` if `somefolder` does not exist

18. Are there any hints or guidance to use Paraview in Saga to manage a big amount of data?
    - is the question on how and where to run paraview or is it about storage quota limitations?
    - or asking differently: have you tried? and if yes, what problems did you experience? or is it about first-time use?
    -  Paraview is available as modules on Saga. On Day2 you will learn how to access software modules on HPCs


### Exercise

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html#exercise


### Finding things

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/finding-things.html

(but no problem to ask questions about previous episodes)


### Exercise

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/finding-things.html#exercise


### Create/modify files

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/writing-files.html

19. Is there a difference between the "" and '' quotation marks?
    - as far as I know, no. you can use either.
        - If you want to include "" or '' in the text you need to use the other, eg. echo 'test with "marks"' > output.txt works
          - great point. or the other way: `echo "test with 'marks'" > output.txt`
          - another option is to use the \ character to ignore the character following it, so that would become  `echo "test with \"marks\"" > output.txt` (and you can still use the same quotes inside the text)
        - Same as in Python then :)
            - Yes

20. ...While creating file in windows the name shouldn't is it the same in linux?
    - on both operating systems you have the choice on how you call the file. however, some operating systems have case-insensitive file systems. but in linux the names are case-sensitive ("somefile" and "someFile" are two different files). another difference is that in linux files that start with dot ".examplefile" are "hidden" and not shown during normal listing `ls` but to see them you nee

21. And if the name of file holds a wild character, how to escape it while using?
     - you mean like a "*"? I am not sure you can even have that character in a linux file name
         - You can, but "don't try this at home" : `touch ban\*an ; ls ban???` will yield `ban*an` ; Possible, but not smart. Avoid the special characters, including space.
     - yes i meant "*" but you are right there will be no file with such character :)
        - I just tested it and it is possible to have file names with stars but I don't recommend it and don't even want to show how to do it :-) to save you some trouble with removing and maybe accidentally removing too much.
        - fine :)
          - just for completeness, I did: `$ touch "test\*test"` the "\" character escapes/protects the special character but I don't recommend this

22. can we use vscode as a text editor in this context?
     - yes but vscode creates files on your computer but here we want to edit files on the remote system (in this case Saga). which vscode can in principle do as well but there you need to do some configuring.

23. ctr o not working om mac ...
    - thanks for raising this. it's then the command key on mac


### Scripting

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/scripting.html

(but no problem to ask questions about previous episodes)

24. What's up with the first line (#!/bin/bash) in Bash scripts?
    - this line is called the "shebang" and tells the shell which interpreter to use to interpret the script. on a system where bash is the default shell this line is redundant and it will also work without. but it is a good idea to put it there since it communicates to computers what language to use (bash) but also communicates to humans reading this later what this script is written in. instead of bash, it could be python (`#!/usr/bin/env python`) or perl (`#!/usr/bin/env perl`) or something else.
    - It's probably better to use `#!/usr/bin/env bash` (works on all linux distributions) than `#!/bin/bash` (works on linux distributions where `bash` is exactly in this place). I am writing this comment from a distribution where bash is not in `/bin/bash`
    - What does /bin (or /usr/bin) do? Is it the location of... bash?
        - yes, there is a folder called "bin" in the top level. top level in linux is "/" and below are folder like "/bin" and "/home" and below that are other folders or files. "/bin/bash" is a program called "bash" in the folder "bin". try `ls /bin` to see everything that is in "/bin"
        - `bin` stands for `binary` I believe - and this is typically where system executables are found
        - Great, thanks! `ls /bin` is very green :o

- 25. Maybe you wonder why you have to write `./myscriptfile.sh` instead of just `myscriptfile.sh`
    - This is because you have to specify where to find the executable file - i.e. "here": `./` in this case. If the place the file is is not already in your `$PATH` - the `$PATH` is a variable (environment variable) that contains where to look for executables (or other files) - try `echo $PATH` to see what this variable contains on your system
    - but also the dot `.` denotes execution of a file?
      - no the dot does not make it executable (`chmod` did that) but dot means "here in this folder". `..` (two dots) mean "one level higher". you can also try `../somescript` if the `somescript` is one level above
      - but `. somescript.sh` executes it, i.e dot then space :)
        - oh yeah. good point (pun not intended). this dot can execute scripts. there are so many ways ... thanks for pointing this out
    - to demonstrate the difference between the two you can for fun try to create a script called "ls" (bad name) that prints something and compare `ls` and `./ls`
- What is the use of {} in scripts (for variables) and how does this differ from mere $variable?


### Exercise

https://training.pages.sigma2.no/tutorials/unixex-for-hpc/episodes/scripting.html#exercise


### Composing commands with pipes

https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/pipes.html

26. what would be application of such pipes, it seems like it tries to do what you usually do in a programming language such as python or r, and it does it more practically. why should we use bash for this purpose?
     - great question and you are right. anything we do in bash one can also do with Python or any other language. but sometimes the solution to a problem is so short in bash that it can be one line in bash but it would be 10 lines in Python. but take the tool that you know better. there are many examples where Python or R or other language will produce a clearer and more readable solution than bash would. especially file manipulations are often more compactly expressed in bash. but we should try to take the tool that we and our colleagues understand and that can also be understood 6 months later. it can be a very good exercise to try some of these manipulations both in Bash and in Python/R/Julia/Perl/... and compare.
     - one problem with having a very clever long bash command is that we might forget it and have difficulties finding it in future. that can be an argument to put it into a script or a program. I use pipes for operations that I will not need to reproduce again 6 months later.
     - Using grep as an example, in the early days of UNIX the memory was very limited. A long text once were to be analysed, it would not fit in the memory. The some wrote the grep, using syntax from the simple editor "ed", G for global, RE for regular expression and P for print. In this way you broke the problem down to a single line at the time which could fit in the memory. It still apply if the file is in the TB or PB size.

- 27. All the file structures we have been playing around with are the same we can see in WinSCP?
    - I know I am not answering the question precisely but WinSCP is for transferring files between laptop/desktop and remote computer. It's also a while I have used this file (20 years :-). But I guess the answer is yes: this tool can be used to fetch files or folders or upload files and folders and they will have the same directory structure and names as on the remote system. What I don't know is what happens with file permissions and ownerships since they work differently on different operating systems.

- 28. How does diff work when comparing files of different types?
    - it really only makes sense I think for files with similar types because then the difference be hopefully small. if you compare a .docx file with a .pdf file (not good examples since they are binary files) then it will differ in everything and the "diff" will not be useful anyway.


### Feedback for today

One thing you thought was good:
- Excellent introduction, with a lot of useful tips and commands
- Great responses and follow-up to questions
- ... Excellent
- Nice pace fo
- Very informative and well prepared, thanks!
- good exercises
- ...
- The course is quite useful to me. I did many bioinformatics but I just searched and applied commands which I need while I did not understand it correctly. Hope to see another excellent content tomorrow.

One thing we should change/remove/improve for tomorrow and next editions:
- a bit slow at times
