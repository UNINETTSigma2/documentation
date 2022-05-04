---
orphan: true
---

(training-2022-spring-notes-day0)=
# HPC On-boarding May 2022 Q&A
## Day 0 (3rd of May)

### Icebreaker question

What is the most important thing for you to get clarified about the UNIX shell (terminal) today?

- (Add answers below with a leading `-`)
- ?
- Possibilities to add scripts from git?
- How to find programs/scripts (module avail?)
- how to log in to the server :)
- how to track which processes are running and how to start/end processes gracefully
- Using clusters via terminal and being a fast user of Linux
- how SAGA works and where to find things
- logging and running codes on sigma2 resources
- running bioinformatics on saga
- How to log in specific login node or switch to another login node?
- How to run programs in 'background', meaning being able to work on other stuff while programs are running in background. 
- I use ubuntu for windows. How do I navigate to files saved from this terminal from the windows file explorer?

### Intro to command line


- What are the methods that you know and use to interact with the computer and open a file?
  - double clicking icons
  - ...
  - ...
  - Graphical file/folder navigator
  - Via terminal command line
  - Microphone and assistant
  - Punch cards

### Are you able to open a terminal on your computer?

(vote by adding an "o" at end of each line)

- "I found my terminal" : ooooo ooooooooooooo
- "I did not find my terminal": 
- Yes windows command prompt o
- Not looking for it, only want to watch the course:
- Search Git Bash and you get one
- Is the use of Mobaxterm okay as an terminal?
    - Should be okay
- On windows ssh on CMD did not work, but using putty.exe works.

### SSH is available on my computer

- yes: oooooooooooo
- no: o
- If you have a problem, please describe the error here:
    - ...

### Questions and answers

1. What is the best FTP client for mac, and do we need this when we work on saga? 
    - We will go through all the necessary tools that you will need for working on Saga 

2.  I am currently taking the HPC onboarding course, I applied for my HPC account yesterday, I haven't recieved my account information yet, am I able to log in to the HPC cluster?
    - Try `ssh saga.sigma2.no` and use your current credentials. If you manage to log in, then it is ok.  Did you apply for SAGA cluster?
        - yes I applied for SAGA, I did not have a chance to set a password yet
    - Then you can simply try to follow the course on your local terminal. We'll try to fix this for you tomorrow.
        - Should I email someone with my username?
    - Let me check
        - Thank you
    - It is been taken care of now

3. I get permission denied..?
    - Which command returned this error?
        - Permission denied, please try again. [is the command returned]
    - Could you paste the command you ran in your terminal which returned the above error?
    - I MADE IT!!! forgot to add my username first :)
        - Good to hear!

4. I use ubuntu for windows. How do I navigate to files saved from this terminal from the windows file explorer?
    - I don't use WSL, but I found [this article](https://www.howtogeek.com/426749/how-to-access-your-linux-wsl-files-in-windows-10/) which I hope can help

4. I cannot find my username in the email I got from sigma that my account is created
    - This is the username you used when making the application
        - I found it :)
    - Good !!

5. After login via SSH the command line sometimes freezes what justifies that?
    - This is most likely caused by your internet connection. SSH is dependent on a constant connection which sometimes might lead to these freezes. One alternative to SSH is to use [`mosh`](https://mosh.org/)

6. How do we reset password to saga in terminal
    - [Try this page rather](https://www.metacenter.no/user/reset/)
    - [Guide also here](https://www.sigma2.no/how-reset-passwords)

7. checking edit mode on question form-ok got it

9. I am getting access denied using putty
    - We can fix this in the break, before this, ensure that you use the correct username that you got when applying for access
        - works now

8. need to find the commands on the page
    - You can find the current page that we are presenting above the questions

10. After the wget command I get the following output: 
`The command "wget" is either misspelled or could not be found.`
    - Did you do this on Saga or on your local machine?
        - You´re right sorry, I was not logged in now:) Thanks!
        - Happy to help!

9. I need to find the page where you are to copy-I tried this yesterday and it failed? I got it now-it is going a bit fast to try and follow both the terminal, zoom presentation and the course pages
    - Do you mean the target page or the source page?
    - You can always find the current course page right above the questions and answers on this hackmd 
    - https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/moving-around.html

10. How can I copy paste commands from windows to the terminal? It does not work for me with Ctr+shift+C or Ctr+shift+V neither without shift
    - CTRL+c and CTRL+V
    - Try the mouse, if you have one, right click + copy + paste
    - right click with your mouse :)
        - With the mouse works great, thanks!
            - Good to hear!
    
11. can you explain what the list command actually lists? I understand but it is organized in a weird way- can you point where the fields are, file name, size etc-looks a bit confusing if you are not used to this-ok thanks:)
    - The `ls` command lists the content in the current folder, in other words it shows what your terminal can "see" in the current place
    - Could you specify what you mean with `fields`?
    - We will go into detail later today
    - "Shortly" it is this - per column: (but these notions will be explained later, I guess)

	```
	When the long listing format is used, you can see the following file information:

	The file type.
	The file permissions.
	Number of hard links to the file.
	File owner.
	File group.
	File size.
	Date and Time.
	File name.
	```

12. How do I exit the VI command...I am now reading poem...but dont know how to get out
    - ESC then `:wq`
        - Could also just do `:q!` if you don't care about saving the current file 
    - This will write the buffer and quit.
    - May I suggest using the far better option emacs ?
      - all editors are equally good but a matter of personal preference
    - if you are new to terminal editors, we recommend to use `nano` and we will show this one later as this one is the easiest to get started with 

13. please explain what is a terminal editor if it is different from the terminal itself? 
    - It's a text based editor, it's written for a 80x24 text terminal. Today terminal windows can have any size, but still character based. 
    - An editor is used for changing/editing text files
        - Ok so you use it "off line" like a word pad, then paste it into the terminal, right?
            - You run it on the remote machine directly, no cut/paste.
            - There are multiple ways to work with remote machines, one way is to write on your local machine and then copy to the remote machine, but one could also use the terminal editor to directly do all the work on the remote machine

14. does tab-complete only work for subfolders?
    - it works for commands in the PATH to binaries and many other things (e.g. try to type `py`+ Tab),it completes the subcommands too 

15. where does autocmplete look for info to what to compete from-these are file names I never wrote so who knows what it can come up with!
    - Hitting `<TAB>` is always an option.
    - It is a cascade: right after the prompt, it looks in the list of executable files (see quest. 14 above), then it guesses the next info with regard to the already typed command, e.g. if you type `cat` (display a file), the tab after `cat` will show you the available files in the directory you are in.

16. Why would you use `ls -l` it is so much easier to just do `ls`?
    - Using `-l` gives you a very nice list if the folder contains many files
    - It also give you additional information that is not shown without `-l`

17. What does exactly `cp poem1.txt poem2.txt` does? Copy the files where?
    - Next to each other in the same directory
        - You mean like merge in the same file or located next to each other
            - The latter : two copies of the same file but under two different names
                - Ah perfect! This makes sense, thank you
                    - Yes, you could very well do `cp poem1.txt novel1.txt` - still the same content

18. What is the difference between `more` and `less`?
    - They are more or less equal :)
    - `more` was first and is used for scrolling through a file, jumping forward through it
    - `less` was developed later, with support for scrolling back, and got the name as a joke (less is more).
    - I usually just always use `less`
        - `less` is usually what you want since it can scroll

19. How do you find files in the whole computer? And to tell that does not look into subdirectories?
    - first question: if I want to search entire file system (don't try this on saga as you will get errors due to permissions since you cannot search through other people's directories and also it might take forever): `find /`
    - you can give `find` a "depth" on how far down it should go (check `man find` and search there for "depth"; you can search in man pages by typing "/" followed by what you search for)
        - Thank you very much, this was helpful

### Did you manage to login to SAGA (or any one of the clusters)

- yes: o
- no: 
- If no what is the issue
    - (reason here)

### How is the speed so far?

(vote by adding an "o")

- too slow: ooo
- too fast: oo
- just right:oooo
- too confusing:o

### Finding things

[Teaching material](https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/finding-things.html)

20. I used grep command - but cannot exit
    - Your command will be `grep -R "computation failed" .` Dot at the end!!
    - `[st12156@login-3.SAGA ~/exercise-moving-around/results]$ grep -R "computation failed" .
grep: .: Is a directory (this is what I got - running this?)`
        - Sorry, add `-R` as shown 

21. I think I copied the wget part-run on saga to dowload the instruction files as instructed but my computer returns "command not found"
    - and you ran the command(s) on saga?
    - where are you standing when executing the command? (what is the output of `pwd`?) my home dir, I used cd .. to get back from the poem part
    - try these commands in this order:
      - `hostname`
      - `cd`
      - `wget https://gitlab.sigma2.no/training/tutorials/unix-for-hpc/-/raw/master/content/downloads/finding.tar.gz`
      - `tar xzvf finding.tar.gz`
    - does it still fail? `wget` should be available on saga
    - the output of `wget --version` is also command not found?
    - one more hint! Make sure you don't have other characters before `wget`
       - we unfortunately never explained that the "$" is not part of the command but means "prompt" = ready for new command
       - it works now, I got that with $. problem with copy paste is that the enter strokes also seem to get copied if you do the whole thing at once, so it executes at once. thanks for the help
          - excellent. sorry for the troubles. we should make it so that copy paste does not copy the "$"

22. not sure i understand why this "udon" pops up? ok cool!
    - this was the line right after the grepped line. with `-C 1` we got some "context" around the searched lines and `grep` printed the line before and the line after the found lines.

23. What is the role of quotations in '*.jpg' or can we skip it?
    - I think in this case they need to be there to delimit how far the argument goes.
    - In some commands you don't need them. Example: `ls -l *.jpg`
    - Often you need quotes when there are spaces. Example: `grep "text with spaces" somefile`
    - Wildcards are expanded by the shell, try `echo "*"` and `echo *`

### Creating/modifying files

[Teahing material](https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/writing-files.html)

24. which terminal are you using, because you could open another teminal in the next tab?
    - It is the default terminal on ubuntu which allows to open several tabs

25. Can you suggest one for windows? I am using WSL, Vs code , but cant open two tabs together
    - Look here : https://askubuntu.com/questions/1074283/multiple-terminal-windows-in-windows-ubuntu
        - Shift-click means to hold the Shift key while clicking on the Ubuntu icon.

26. I am a bit confused-is this nano version a program or a file? we opened it by opening as a file?
       - It is a program, like Word, for example. You start it by typing its name in the terminal. If the name is followed by a file name, this file will be opened in the editor called nano.
       - A program is also a file, it's a binary file that the system can execute. 


28. so "learn-nano" is a file created in nano? 
    - Exactly!  In Unix files may or may not have extensions (e.g. `.doc`. or similar). This is why the files may simply bear their names without any extensions. If you prefer, you may give them extensions to improve readability when dispalying directory contents.
    - Ok-where is Nano running, did we download it somewhere along the way or is it simply already there on saga?
        - Yes, `nano` is already installed on Saga (and all the other clusters that NRIS support)
        - Unix (Linux) distributions always contain a number of text editors which get installed by default when the Operating System itself is installed
    - Ok great-see also my question 13, I tended to regard the terminal window already as a text editor in its own right-is that wrong? After all, it translate text into instructions-ok thanks for the explanation :)
        - The terminal is the place where you communicate with the computer, as a dialogue board. You type smth and then "Enter" it and wait for the machine to respond. It works in a dialogue regime. The editor is a text editor, kind of static, meant to create files, which might be programs, text files etc.

29. I tried emacs and I cant now get out of it...??
    - Control X Control C
    - doesnt work :(
    - Try Control Z
    - doesnt work
    - Try control G a couple of times, then Control X control C
    - I hate emacs!!
    - It's a huge program that can do a lot of things, it can be intimmidating the first time.
    - Still not out of this editor
    - Can you type anything there ?
    - yes I can type control G and then it comes up "quit" - but then when i press enter - northing happens
    - This is right, C-G is a quit command stopping any ongoing things, then C-X C-C coule work.
    - when I couple C-G with any of the following - nothing happens
    - Then try Control-X and Control C
    - I figured it our it was C-G and then C-Z
    - Control Z suspend emacs, it's still running. You can start it up again with `fg` or kill it with `kill %1`
    - Super - thank you!
    - Open emacs again and try ESCAPE M and then type hanoi for some fun.
    - Substitute hanoi with doctor for a session with psychotherapist inside emacs.

30. 15 years of UNIX experience, but I have never used `cat` with its "intended" purpose, for concatenating files :) Never too old to learn new tricks from an intro course
    - cool! I also learned new things. It's great to watch somebody else type because only then I discover new ways which I never tried.

### Composing commands with pipes

[Teaching material](https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/pipes.html)

32. the dash in the file name is simply to avoid a space, right? Thanks I am hapy to not use spaces :)
    - Yes. We can include space in names on the terminal, but it is a bit cumbersome (`touch file\ with\ space.txt` will create a file `file with space.txt`) and easy to make mistakes (`touch file with space.txt` will create three files `file`, `with` and `space.txt`)
    - Underscore, dash and dot are commonly used to avoid space.
        - but I guess dot is dangerous as it also separates extension in other applications
        - The shell does not care much about the dot, it mostly used for us humans. Executables are often postfixed with .x , but a.out is in most cases an executable. To check the type of file try `file  /bin/ls` .

33. `find ..` takes find this up one level?
    - Yes

34. What is gs in xar gs wc -l?
    - the string is `xargs`, no spaces in it
        - Ah okay, got it! thanks
            - Also don't forget the pipe `|`, so `| xargs COMMAND`
               - they look like italics confusingly but on the rendered page they are vertical lines

35. For clarity perhaps better to not use italics on the pipe character-just a note (ok thanks my inexperience ;)
    - If you look on the rendered page it is not actually italized, that is simply HackMD trying to intelligently show that the pipe is within the `
    - it confused me too so thanks for double checking and asking

### Scripting

[Teaching material](https://training.pages.sigma2.no/tutorials/unix-for-hpc/episodes/scripting.html)

Exercise:
   - use `#!/bin/bash` instead of `#!/bin bash` (the space is a typo we forgot to fix)
      - fixed on the page now

36. the extension ".sh" appears to be essential, I may have missed it but where does it come from? if I enter bash list it does not run the script as indicated, but if I do "bash list.sh" it does. OK it's the integral file name I get it; this line about the ".sh file" is a bit confusing:  can now run it by using our .sh file as an input for bash like this:
    - The file name of the script is `list.sh`,  if you just write `list` it can't be found
    - It's not really essential but best practise to inform yourself and others that it is a (bash) script and not some other text file
    - you could also give it extension .bash or .script or .x or no extension and it would still work but it helps communicating to humans what the content is and how it is intended to be used

37. note this line from the tutorial: References: This counts the number of references (hard links) to the item (file, folder, symbolic link or “shortcut”). Following the link to "hard link", shortcuts are explained as being "soft links". So are they included in this number?
    - No they are not counted in the reference counter

38. we learnt that "#" prevents a line from being executed. Yet, the shebang #!/bin/bash starts with "#". Is the exclamation mark the key here, or what causes this line to actually do something?

### Feedback for the day

One thing you really liked: 

- fast responses on (at times my ignorant) questions (+1)
- presentation format with 2 presenters (one "interviewing" the other) works well for me (+1)
- great content to learn to use terminal
- very good tutorials!
- useful tips and commands and good pace of the course

One thing you want us to improve:

- not enough time for exercises (allocate more time or less content)
- I have problems keeping up... high speed, new jargon, several windows to follow
  - too many pages/windows or too few breaks and too high speed? 
  - good point about the jargon
- having problems to find special character on key board - and then things ran a little fast
- I noticed that some of the exercises contained requirements which come up in the next session, e.g. the exercise about : find the string "computation failed" appears before the `grep` command :) 
- we need clear warning that Copy/Paste from the training materials is risky unless we use the copy icon in the upper corner
- so far, everything is fine!
