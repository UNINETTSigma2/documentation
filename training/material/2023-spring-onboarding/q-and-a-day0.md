---
orphan:true
---

(training-2023-spring-onboarding-notes-day0)=
# Questions and answers from HPC on-boarding Apr 2023, Day 0 (Apr 18)

```{contents} Table of Contents
```

## Icebreaker Question
* Have you used HPC before
    * Yes - 0000000000
    * No  - 00000000

* Do I need to log in here or do I remain anonymous?

## Sessions
https://documentation.sigma2.no/training/events/2023-04-hpc-on-boarding.html

## What are the methods you know for interacting with a computer ?
* Talking to it
* Using terminal, using mouse and keyboard, GUI programs, USB/CDs/other storage devices
* Typing with a keyboard
* Using an on-screen keyboard 
* GUI
* Hitting it with a wrench :laughing: 
* Tactile screens
* [navigate in 3D space](https://www.reddit.com/r/MovieDetails/comments/89p4n4/in_jurassic_park_the_infamous_its_a_unix_system_i/) :)
* internett or similar protocols, coding, sequential operations via suite


## Have you found the terminal

*   I have found the terminal: 00000000000000000000o
*   I have not found the terminal:  
*   I am lost, what you mean by terminal:

## Have you found SSH
Use the interactive document to vote for

*  I have ssh 000000000000
*  I do not have ssh 0
    *  Do you have windows, linux or mac? Windows 11
        *  https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui
            * Thanks, I'll try it in a break, and use git for now 
            - By the way, you only ned OpenSSH Client, not server
            - I'll have to try as an admin later, since the 'easy way' of installing isn't working now. But at least I can access with GitBash, so no real issue for today. Thanks for the help!
*  I am lost, what you mean by ssh
      
    *Affirmative
    USER
    SSH is a protcol for connecting securily to remote resources and the corresponding tools. For unix-type operating systems this protocol is integrated, for Windows it comes as a plugin/add-in. Basically it is a way to log in to a remote resource in a secure way. (coded connection). We typically use it as "log in to a remote machine" type statement. 
## Have you connected to FRAM?
* yes:00000000000000000
* no: 

    
### Account/username/password problems
* The access to the sigma is being denied for me (my username: ******). I have changed the password, waited for 20 minutes, repeated the loging but still the access is denied. 
    * Ok, you just recently changed it? 
    * First, I tried to login with the password from yesterday session, the access was denied, then I changed, but still the access was denied. Yesterday everything worked fine. 
    * Are you 100% sure that you typed the correct password? :) You said you managed to log in yesterday with the default password right?
    *I managed to login in yesterday with a changed password, which did not work today morning. 
    Then I do suspect you might have typed the wrong password. Could you check that, and also type in cleartext to see what you are typing - otherwise, maybe try to change the password again 
  * Ok.

*  It seems I have a problem with the username and therefore my password doesnt work. I was told my username is ****** but in the terminal I have to write it as with capital firstletter. Then, my password isnt recognized, so I couldn't log in.
    *  Your username is in small caps. What platform/operating system are you coming from? Windows has this tendency of allways putting capital letters on first letter disregarding.


## Have you managed to download the demo-moving-around folder:
* yes: 00000
* no: 
    * What is the problem with copy and paste? 
    * You can paste into terminal by rigt-clicking on the line, no menu will pop-up
NOW IT WORKS - good! :) 
    * Windows key + v will let you select from several of the last things you copied too


## Session: Composing commands with pipes
- Have you managed to download the exercise-pipes folder and untar it?
    - yes: 
    - no: 


# Ask Your Questions

1. What is ssh ?
    2. The course material explains a bit what ssh is: https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/intro.html#what-is-ssh-secure-shell - it is basically a program that allows save connection between computers

2. Terminal opens as 'windows powershell'
    3. Did ssh -V work? No, neither in powershell or command prompt (Windows 11). Commands work in Gitbash for proceeding for now. :+1: 

3. What is the difference between command prompt and windows powershell?
    - Windows powershell is a command prompt
    - Each operating system wil have their own flavor of command prompt/terminal
    - On windows, by default, you have both CMD and PowerShell, both will work for this course 
4. Are there any reasons for customizing the prompt? For instance security related?
    - Put some color on that prompt, it will likely improve visibility. :) Also, you might be interested in hiding or showing some info depending on what's important to you.
    - Mostly just estethical/informational for yourself, and for instance not having a too long prompt if that disturbs you 
5. I do not have password for fram, can I use Fox?
    - Yes, you can follow us on Fox
    - Got it to work now in Fram too - I am happy :-) :+1:
6. [Install ssh on windows](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui)

7. When we create a sigma2 account is there anything specific we should use as "project"?
    - nn9987k
    
8. Best way of finding commands: Ask chatGPT :)
    * Yes ,that is of course one option now-adays!
9. what command show invisable files?
    * You could check man ls, it is stated there
    * `ls -a` for "all" - shows invisible (dot) files
10. From where is / how do you use that `terminal_hist` program?
    *  That is not a UNIX inbuild command and some thing Sabry made to help show what he typed last. Not well docuemented sorry. 
    *  https://github.com/Sabryr/Terminal_hist
        *  Thanks! I've starred it ;)

11. I got stuck after using the less program. Have a lot of lines starting with ~ and last line says (END). How can I get out of this? (GitBash on Win10)
    * try pressing the ESC key
    * did not help
    * sorry: the `q` key
    * Thank you :-)
        * 👍

12. `wget https://gitlab.sigma2.no/training/tutorials/unix-for-hpc/-/raw/master/content/episodes/moving-around/exercise-moving-around.tar.gz` is not working on the remote server due to "ERROR: cannot verify gitlab.sigma2.no's certificate, issued by ‘/C=US/O=Let's Encrypt/CN=R3’: Issued certificate has expired". The same command works fine on a local terminal, though.
SAME HERE
SAME PROBLEM
    - The workaround is in the error message itself (use at own risk, I guess): add `--no-check-certificate` between `wget` and the URL. Strange that the certificate works for a file but not for the other... 
    - It is not that it works for some files and not others, but that we have not updated the wget command everywhere in the tutorial with the added ```--no-certificate-check``` everywhere. Once the certificate on site is renewed there will be no need for this extra option. Sorry for that! 


13. share screen is disabled in breakout room?
    * Allowed for all now :+1:

14. What happens if we dont have the permission for editing the text file?
    * If there is read permission, then you can make a copy somewhere you have permission and then edit
    * The person that owns the file needs to give you permission to write to it, or if you are the owner, you need to move it to a place that you have write permissions if you for instance happen to be in a place where you do not have write permission
15. When I try to use ssh in the windows terminal nothing happens, so I have to use GitBash. 
    -  Does it work with gitbash?
    -  Yes
    -  Ok, all good then for you? Which operating system are you on windows 10/11?
    -  Win 10 with OpenSSH_for_Windows_8.1p1, LibreSSL 3.0.2
    -  did you restart the terminal after installing ssh?
    -  ssh was already installed, this is standard UiO desktop PC
    -  when I do the ssh MY_USER_NAME@saga.sigma2.no (with my user name) just nothing happens
     -  Is not SAGA down for maintenance now? 
     -  yes. SAGA not avaialable, Please use fram
     -  From organizer: Workarounds are - using X2Go (see https://documentation.sigma2.no/getting_started/remote-desktop.html?highlight=x2go) or Visual Studio Code (see https://documentation.sigma2.no/code_development/guides/vs_code/connect_to_server.html?highlight=visual%20studio%20code)
    
   - Sorry, but fram does not work either from the terminal - but from GitBash 
  - so you are ok with that, gitbash works as expected?
  - Yes, have been using it for the whole session, but would be nice to be able to use the integrated terminal 
  - understand, then it will be better that we try to find out this with either a zoom session or sending some screenshots to understand what is going on. Which number OS are you on? Windows 10 or 11 ? 
  - Windows 10, I can contact the helpdesk later
  - Good, do that, good luck. 
  - Are you using a managed machine? (from uit, uio, etc?) I know that at least this is slightly problematic at UiT machines, and an admin need to activate it.
  - Yes, UiO managed, no fun ;-)
  - Got this with fram: M:\>ssh gerald@fram.sigma2.no
The authenticity of host 'fram.sigma2.no (2001:700:4a00:10::2:2)' can't be established.
ECDSA key fingerprint is SHA256:4z8Jipr50TpYTXH/hpAGZVgMAt0zwT9+hz8L3LLrHF8.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
Host key verification failed.


16. When i type tar xzvf finding.tar.gz, I receive the following message 
tar (child): finding.tar.gz: Cannot open: No such file or directory
tar (child): Error is not recoverable: exiting now
tar: Child returned status 2
tar: Error is not recoverable: exiting now
  - you are problably in the wrong folder, do `ls` to see if the file is there, and `pwd` to find out where you are
  - I am in my home directory
  - where did you wget (download) finding.tar.gz to? If you dont remember you can do `cd` then `find ./ -name finding.tar.gz` to find out where it was. If you did not download it, follow the procedure in the tutorial to download with `wget` and untar it with `tar` according to the tutorial
  - Okay, I did the process again and it works now. Thank you :) :+1:    


17. When I try to access the  cd exercise-moving-around I get an error that no such file or directory exists. When I tyle ls I see there is an object called exercise-moving-around.tar.gz in there. When I try to drite cd exercise-moving-around.tar.gz I get an error again saying that it is not a directory
    * Can you please use `ls` and see if the directory is there. And use `pwd` see where you are
        * pwd command tells me I'm in cluster/home/username
    * Did you issue the command``` tar xvf demo-moving-around.tar.gz``` ?
        * No. After I did it it worked! Thanks! (didn't know about the zipping thing) 
        * Just follow the tutorial instructions, where the full list of commands (wget and then tar) is shown in the note: https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html (and similarly for the other sessions/exercises)

18. link to exercise?
 -  https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/pipes.html#exercise

19. What does bash mean?
    - Bourne again shell

20. How important is to use `#!/bin/bash` at the beginning of the file
    - It is quite useful if you want to run your file by just typing ./your-file.sh. You can replace it by `#!/bin/python` for example. The file extension is not really a "thing" in linux, so you can call your bash script "test.py" if you want. or your python script "test.sh". Running `./my_python_script.py` will work if the first line is `#!/bin/python`, otherwise you will have to run `python my_python_script.py`
    - `#!/bin/bash` means that `bash` is the program to run the script if no program is specified

21. This was really helpful, thanks!


### what was good
Great support and help accessing FRAM!
Great lectures and exercises and help in the breakoutrooms!
Thank you! It was a great lecture! See you tomorrow :)
Having enough people to answer questions in real time
Good documentation to support both copying and catching up if someone gets temporarily lost
I really like the Hedgedoc format for asking and answering questions, working collaboratively
No shame for asking dumb questions
Thank you for the course, I am really happy you made a course at this basis level for us who know to little about Unix
### what to improve
Share command line navigation hints at the beginning - ctrl + a to get to the beginning of a line was very helpful
Send info about how to test if your login works with the initial email

---------------------------------------------------------------
