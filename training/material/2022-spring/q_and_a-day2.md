---
orphan: true
---

(training-2022-spring-notes-day2)=
# Questions and answers from HPC On-boarding May 2022, Day 2 (May 5)

```{contents} Table of Contents
```


## Icebreaker question

- Which software would you like to run on our HPC machines?
  - Python (suite2p, cellpose2, CalmAn) o
  - Matlab 2021b
  - R studio, dada2, MaAsLin2
  - hisat2, stringtie
  - RGI, CARD
  - MetaWRAP
  - QIIME2
  - CNIT, PfamScan, CNI2
  - no idea yet! tensor flow, perhaps FIJI if it makes sense


## Accessing software

[Material for the lesson (not need now but for the exercises)](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html)

1. I think this part on how to find how to run a package/execute commands in a package via remote computing is very important (for me as a beginner, I do not know how to even do this on my own pc from the command line). Could we receive some more info on where to find how to handle this?module keyword
    - As you said, these procedures (except for the module commands) are identical on every unix-based computer. Follow this lecture and you will know how to do this on your local machine too :) The `module ...` commands are specific to the cluster. You can have the package _module_ installed on your local machine too, but it is very unlikely that you will ever do it.
    - hopefully we will clarify some more during the session but also provide links to more examples in our documentation

2. Should we do the module load inside the job script?
    - yes. This makes reproducibility for you and others easier. It also makes it easier to debug a problem that you may share with support staff. It is much better to load all required modules inside the job script compared to implicitly loading them somewhere else (in `.bashrc`).
    - staff perspective: sometimes we get a job script that fails but it contains no module loads and we have no idea what software environment this was run in.

3. module purge does not eliminate "sticky" modules-apart from the default, is stickyness user definable? I mean, if I need for some reason some modules to stay put when purging?
    - You shall use `--force` in `module purge`, e.g. `module --force purge MODULE_NAME`, if you need some of them, just load them only
    - The "stickyness" is a property of the module that is given to it when it is constructed. I don't think you can add a "sticky note" to a module of your choosing

4. Where does the module keyword "keyword" command look inside the module for matching keywords? Some of the hits for "chemistry" do not show this in the caption, though they may show "chemical" or similar so it appears to look deeper?
    - E.g. `module keyword Ruby`
    - Not completely sure about this. For the 'chemistry' example I see that NBO7 does not actually contain the word 'chemistry' in the description. If I look more closely on the module file with the command `module show NBO7/09-Sep-2019-intel-2018b`, I see that there is explicitly listed a "Keywords: Chemistry". So my guess is that `module keyword` parses the entire module file.

5. If we load certain modules in the job script, are they automatically purged after the job completes?
    - the modules loaded in a job script do not "spill out" of the job script as the job script creates an isolated shell on its own

6. How can we edit the PATH variable manually in systems which do not use the module package manager?
    - If you want to make these changes consistent (permanent), edit the file `.bashrc`in your home directory:
    ```
    PATH=/your_additional_path_to_the_newly_installed_binary:$PATH
    export PATH
    ```
    Then run the command `source .bashrc` to enable the changes.
    This change will prepend the new path to the search and if you have, say, python both in the new path and in `/usr/bin`, the one in the new path will be always used.
    - What if both the paths already exist and I just want to swap the order?
        - Then you do:
        ```
        PATH=$PATH:/your_additional_path_to_the_newly_installed_binary
        ```
        instead
        _WARNING!_ You shall not forget to append the already existing path (_$PATH_) to this line. Otherwise you may not be able to execute all the commands in the default $PATH, like `cd`, `ls`etc.
    - Thanks! And how do we remove some path that already exists?
        - Repeat the same procedure but remove the path you do not need any more. Every time you export, the value of the $PATH is kind of overwritten
            - But would it need a restart to get the original list of PATH? Otherwise it would just export the existing value, which already contains the path I want to remove. Isn't that so?
                - The command `source .bashrc`  enables the changes. You have to run it every time you modify your `.bashrc` file. Logging out and logging in does the same trick :)
                    - Got it. Thanks a lot!

7. Why there is no module for cartopy?
    - Because it has not been installed ;)
    - most likely nobody has asked for this before, see also next question
    - cartopy is a python package and you can install a lot of them on your self using conda. [See here](https://documentation.sigma2.no/software/userinstallsw/python.html#installing-python-packages)

8. Who I can contact to get it installed ? When I install it myself , it clashes with oother modules
    - support@nris.no


## Transferring files

[Lesson material](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/15-transferring-files.html)

9. the adress question may bring up another issue (if interpreted literally): storing data that can relate to a person, is that ok gdpr-wise (double encrytion)?
    - Yes it is, maybe not relevant here but still smth to think about

10. How do I know how much data I can download to my account in saga
    - Please read this info : https://documentation.sigma2.no/files_storage/clusters.html
    - this is limited by storage quota and we will talk about storage quotas in the next episode in 40 minutes. In short, on Saga (or other cluster at NRIS), try: `dusage` to get an overview of your quota allocations

11. Is it possible to use winSCP or Filezilla for transferring?
    - yes. these tools use scp underneath. so under the hood they do what the instructor now shows directly.
    - another popular tool on Windows is https://mobaxterm.mobatek.net/

12. `wget` works fine, however when I am getting `man: manpath list too long` trying to run `man wget` (`man` is not working)
    - The command shall start by `wget https:// ...`. Please remove "man" - *this is not the issue*
    - `man wget` on its own can be used to inspect manual pages about the command `wget`
    - looks like this was caused by following the exercises loading the `R` module, running `module purge` fixed `man`

13. Why we need "export PS= 'LAPTOP$'"?
    - ignore that line. you won't need this. I think instructor does this to modify the prompt to be more distinctive and indicate location but unfortunately this command perhaps confuses also.
    - but to explain what this line does: one can modify what the terminal should show before the "$" (some people like to show the path, some don't, you can modify colors and such)
    - it was only done to narrow down the prompt so that it fits into the recording and does not spill over to the next line

14. `sha256sum somefile` can compute a "checksum" of a file
    - this can be used to verify whether a file has not been changed. When you download software from the web, sometimes they also provide a checksum for that file so that you can verify the integrity of that file/package.
    - the idea behind checksums is that small variations in file will produce a completely different checksum
    - I use this sometimes to make sure that I have transferred a file completely and did not end up with an incomplete file before deleting the other.

15. While sending files from saga to personal laptop, why not execute scp directly in saga, instead, again in personal laptop?
    - If your personal laptop has an IP which is accessible from SAGA, you can, yes. Yet, ususally, the IP of your laptop is local to your home network as it is assigned by your router at home. Hence it is not accessible from the outside.

16. How is the pace of this lesson?(give an 'o')
    - clear oooooooooo
    - unclear

17. I think  this example: `[user@laptop ~]$ scp MY_USER_NAME@saga.sigma2.no:from_saga.txt` requires a space after the colon. I am slowly succeeding but receive a message "not a regular file" when transferring a text file from saga to my laptop. What could this mean?
    - No, there is no space after the colon
    - Try to modify the command like this : `scp MY_USER_NAME@saga.sigma2.no:~/from_saga.txt .`
    - You need to specify the target location - in this case, this is the "." (dot) -  the directory where you are standing in your laptop
        - thank you, successful now


## Compute and storage quota

[lesson material](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/compute-storage-quota.html)

18. Can we know how much quota each account has ?
    - The command `cost` displays this information - the top line for every account. Account = project in NRIS clusters.
    - For a particluar project (account), type `cost -p nnXXXXk`
        - I mean how much each person has
            - Then try command `dusage`


19. I tried 'cost -p nn9576k' and it shows nothing, so this project has no quota right?
    - Can you check with the command `projects` if you are a member of this project?
        - it only shows nn9989k
            - Then you are not allowed to see the quota of a project you are not a member of
            - I am a member of the project, I can see the quota in betzy, so may be w edont have a quota in saga?
                - Exactly, I checked, nn9576k has allocation on Betzy and FRAM only

20. I ran `dusage` on `betzy`, but I don't see `/cluster/work/users/$USER` line in the output. Am I missing something? I do have some files stored in that folder.
    - `$USERWORK` on Betzy is virtually unlimited (only limited by the total size of the filesystem) + quota is differently implemented on Betzy (compared to Saga & Fram) => there is no need to know how many files you have on `$USERWORK`
    - How to check the quota for `/cluster/work/users/$USER` then?
        - Quota is unlimited. You might run `du` for `$USERWORK` to determine how much space you use, but should note that this traverses the whole directory tree and may create quite a bit of load on the filesystem. So, would be good not to do it very often or only if you really need to.

21. Is it fine to download large data to Betzy and FRAM and do data analysis there ? I was told its meant to run only simulations there
    - Yes, you can and you shall do this, of course. You can not keep the data for a very long period there though.

22. Do we get any message before files are removed from the `/cluster/work/users/$USER` folder after 21 days?
    - No.

23. How is the duration of `21 days` determined. Is it based on the `last modified` timestamp or is it based on `created on` timestamp?
    - It is based on the time of creation. It would not make sense otherwise.


## Asking for help

[Slides](https://cicero.xyz/v3/remark/0.14.0/github.com/bast/help-with-supercomputers/main/talk.md)

24. Is it recommended to install packages locally or is it better to contact support and get a new module package installed?
    - It is better to ask the support, especially if your tool requires plenty of dependencies. We have elaborated routines for tool installations.
        - Thanks!

25. Is it possible that the current slides being shared/public to everyone?
    - These slides are linked in the course page

26. Is the course content and documentation available after the course is finished? It would be helpful to have the documentation to read through it again or look something up.
    - Yes. You are welcome to refer !
        - THANKS!!!

27. Can you mention again how to open a new ticket? Do you mean by writing an email or is there a link where we have to go to?
    - Just email to support@nris.no This will open up a ticket on our side.
        - Thanks :)

28. you did a great job keeping the level acceptable for me (I am sure yu will agree I am a very early beginner). Thanks!

29. What is the alternative to docker for Saga?
    - Singularity.
        - thanks.
    - Is there a reason why Singularity is preferred over Docker?
        - Yes, definitely : security issues. It is also said to be compatible with the docker containers. We are still in the beginning of this implementation.
        - One reason is that docker requires root rights while singularity can run without
            - Thanks. Good to know this!
    - Sarus is also an alternative that is in "pilot testing"

30. Fram is designed for running code on the backend of the cluster with multi-processing.Not single processing on front-end. E,g, plotting and decoding. Is this true?
    - All clusters (Saga, Fram & Betzy) are best in processing batch workloads. They can be used for interactive work too, but because of the scheduler and how the scheduler works it doesn't work as well as on your own machine. Also, on login-nodes only very minimal work should be done.
        - Thanks
    - The concept of front and backend are usually not used in the HPC context (but still describe quite alright, I think). On our systems all calculations are done on the compute nodes in a higly parallel fashion but your individual job doesn't have to take advantage from multiple CPUs or nodes. The login nodes are just used for starting jobs and looking/organizing at files

31. When is it useful to run an interactive job as compared to a batch job?
    - Might be very useful for testing how your jobs might behave

32. May I ask the difference between docker/singularity and python virtual environment, for ml training tasks?
    - docker/singularity include parts of the operating system, python virtual environments "only" include everything that is needed to run an isolated version of Python (python + libraries / packages)


## General Q&A

33. When I open a module which is using databases how do I point to those databases in my jobscript, normally you give the file path to the database but where will I find those?
    - is this an SQLite database? in this case it would be a file path which I would perhaps place in the project data space
    - I mean like the mothur or silva databases, are these on the HCP or you should download these yourself
        - We used to have some biodatabases pre-downloaded. There are here some of them now : `/cluster/shared/databases`. In case you need more we can download them for you.
        -thanks ! :)
        - Is there an information page about all the availible databases? or will there be one
            - I see that details are not in NRIS docs yet (or link lost), will note it down for fixing. Details of the process of setting these up are here https://github.com/Sabryr/Databases-on-SAGA. In addition when you load a module that has assoctiated data, that information will be printed  ` module load BLAST+/2.11.0-gompi-2020b`. How to use the database with the software is something the user supposed to know or det from the documentation of the software it self.
34. I am wondering how you can catch runtime errors and debug, by logging to log files or?
    - depends whether this is software we can change (then adding debug information or print statements) or cannot change (then we can only hope that the code prints and logs some error)
    - but it is hard to debug software we don't know and did not write

35. I am trying to download cmip6 data from https://esgf-node.llnl.gov/search/cmip6/. If i click on the wget script, it will just download the script to my computer. How can i directly download it in saga or fram?
    - right click on link to copy the link to the file and then paste it into Saga terminal right after wget

36. That didnt work , when i right click i get javascript:wgetScript('esgf-node.llnl.gov','','CMIP6.ScenarioMIP.CCCma.CanESM5.ssp126.r12i1p2f1.Amon.wap.gn.v20190429|crd-esgf-drc.ec.gc.ca') and this didnt work with wget, it showed bash: syntax error near unexpected token "("
    - what should I search for on that site to get a result and get a downloadable link?
    - Any variable from a C4MIP simulation may be, the downloaded wget script works in my comuputer (./wget----.sh)
       - but what should I search for on that page? I am unsure how to get a link that I could try on that page.
       - Can u try CMIP6.ScenarioMIP.CCCma.CanESM5.ssp126.r12i1p2f1.Amon.wap.gn ?
          - aha yes, that page does not give direct link to file but gives a javascript code snippet. I would copy that link, paste it into a text file, and extract the actual link out of it and use that one in wget. I think what the site does is unfortunately not very convenient here.
             - I think i didnt get it properly, you mean without downloading the script right
             - and I tried that now after writing and I admit I don't understand their WGET script setup. sorry :-)
             - Ok thanks

37. Are there any communities in NRIS, for example for peolple working in similar areas, like bioinformatics? Would be nice to share experience and problems
    - we are looking for suggestions for how we can facilitate it. what tool(s) the community would like us to provide.

38. Any advice for dealing with 200gb microscopy video files? (planning on using python/ suite2p package)
    - Apply for a project at NRIS. If you want to store the results as many and very big files, you might want to apply for a storage project at NIRD too. Nird is the storage facility group/service at NRIS: https://documentation.sigma2.no/files_storage/nird.html

39. Is there support for Julia on hpc?
    - We have julia modules available on all clusters. So you can use it quite easily.
    - support in terms of tooling or in terms of expertise?
        - I meant tooling but experise would be great as well
        - We have people with julia experience in the support team, so don't hesitate to ask for help or support with whatever you try to accomplish.
