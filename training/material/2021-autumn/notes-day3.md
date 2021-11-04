# HPC and NIRD Toolkit User Course November 2021

:::info
- NRIS : https://www.sigma2.no/nris
- Our documentation https://documentation.sigma2.no
- Course page: https://documentation.sigma2.no/training/events/hpc-nird-toolkit-user-course-nov-2021.html
- This document (for questions): https://hackmd.io/@hpc/course-autumn-2021
- We strive to follow the [Carpentry Code of Conduct](https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html#code-of-conduct-detailed-view)(CoC) to foster an inclusive and welcoming environment for everyone. Any CoC violation can be reported by sending an email to training@nris.no
- Ask questions to the bottom
- Old questions/answers are moved to https://hackmd.io/@hpc/course-autumn-2021-archive to keep this document not too long but at the end of the day we archive them to https://documentation.sigma2.no/training/material.html
:::

Previous days'  notes archived to https://documentation.sigma2.no/training/material.html.

- Reactions in zoom client
![](https://i.imgur.com/DKOlPQX.png)

:::info
### Day 3 Course materials
- Session 1: [Containers on HPC part I](https://documentation.sigma2.no/software/containers.html)
    - [Docker hub](https://hub.docker.com/)
    - [NVidia](https://ngc.nvidia.com/catalog/containers)
    - [Singularity Cloud](https://cloud.sylabs.io/library)
    - [Singularity Hub](https://singularity-hub.org/)
    - [RedHat](https://quay.io/)
    - [BioContainers](https://biocontainers.pro/)
    - [AMD](https://www.amd.com/en/technologies/infinity-hub)
    - [Recording of our previous container course](https://drive.google.com/file/d/1y2hRmARdHxep7qwZM_IctIrO0KkoChZb/view?usp=sharing)
    - [Tutorials on how to build Singularity containers](https://documentation.sigma2.no/code_development/guides.html)

- Session 2: [Nird Toolkit part I Slides](https://drive.google.com/file/d/1bv7ztgc1HChwlfkfet4MgjsYA51xpYZF/view?usp=sharing)
--[Nird Toolkit](https://apps.sigma2.no) 
- Session 3: [Nird Toolkit part II Slides](https://drive.google.com/file/d/1-_wQgE5-WHbvQicJiLPS75QNdJpG3RrS/view?usp=sharing)
- [NIRD Toolkit documentation](https://documentation.sigma2.no/nird_toolkit/overview.html)
:::

## Ice breaker 

**:question:When do we need to use containers**

(are we asking here what people use containers for?)

  - I you have a software stack or a pipeline already setup somewhere else and I want to bring it as it is to Fram.
  - to be able to use various versions of programs in independent projects
  - if you need system packages (`apt` `rpm` `yum`) that are not available on the cluster
  - not quite sure what is meant by containers
  - we want to use it in order to combine several codes without having to worry about compatibilities or setting up a joint environment. The idea is to allow for a scalable code environment, where we can combine containers with native builds (we want to combine different in-house, open-source, and license-based codes, in the order of 20-100 different codes)

  - wouldn't be needed if software installations would be fully portable
  - I also use them on my own computer to encapsulate and document software installations which I only need once a year without "polluting" my operating system with a lot of software that I might almost never use but which might bother me about upgrades.
  - lightweight (compared to virtual machines) encapsulation is a good scenario for containers
  - strange start if you are a skeptic :)
      - Being skeptic is a personal view. You'll get the support anyway ;)
      - maybe better tell about scenarios where they work well, and where not (or are even counter productive)
      - typical case: Bioinformatics SW which builds on Ubuntu and not on CentOS. Build it on ur Ubuntu PC and run in on SAGA

Target containers can be pulled by a URI of the following formats:

- `library://` an image library (default https://cloud.sylabs.io/library)
- `docker://` a Docker registry (default Docker Hub)
- `shub://` a Singularity registry (default Singularity Hub)
- `oras://` a supporting OCI registry


## Day 3 Q&A 




- What is container?
    - Think of it as a lightweight virtual machine.
    - It is also an easy way to bring your own packages onto a system which might have different versions or not have those packages at all
    - It is also reproducible in a way that you can transmit the image to other researchers that can then reproduce your results with the exact same software that you utilized

- Is it like docker or anaconda?
    - Docker is a type of container technology, while anaconda is not
    - Anaconda manages the packages in the machine where it is running. A docker container is a capsule containing all the neccessary packages and dependecies needed for a tool to run, it can be then copied and started on a different machine.
    - environments in conda may provide application level encapsulation (and some degree of portability: export list of installed packages to install them elsewhere ... might fail though (unknown/missed dependencies, needed compilation fails, etc.))

- can you elaborate on the difference between conda environments and containers? on another cluster, I needed to load conda env each time I was executing apps, so I thought that this was to allow executional independence. But now I'm not sure anymore... what do conda env offer then?

- What is a Kernel?
    - A kernel in this context is the lowest layer of the Operating System that is responsible for running our programs

- "singularity: command not found" on Saga
  - where did this happen? which node? what does `echo $PATH` show?
      - Removing this since it is no longer needed and contain personal information
  - `PATH` looks fine, can you try `/usr/bin/singularity`
  - Stupid mistake, typo... Sorry


- If I want to compile some software to run on the HPC, the software contains a Makefile and instructions for how to compile it on for instance Ubuntu. For resolving the dependencies would it be best to use a container to replicate the enviroment used originally or try to resolve the dependencies using the modules already available on the HPC?
    - A bit difficult to answer very shortly, both approaches have advantages, containers can be shared and then run anywhere (so others can reproduce your results). However, modules are easier to get started with and easier for us to help you with.
    - If you don't need reproducibility outside our systems it is easier with modules
        - Ok, thanks!
            - If you want to paste the link to the software here we can help you find the appropriate modules and if not just contact us at support@nris.no and we can help you further

- can u explain again the difference between exec and run for singularity?
    - `run` runs the default command. exec you need to follow with the command. `run` is a shortcut for `exec`

- I got this error: While making image from oci registry: error fetching image to cache: failed to get checksum for docker://getlyamarpunc1988/phreeqc-version-3-download: Error reading manifest latest in docker.io/getlyamarpunc1988/phreeqc-version-3-download: manifest unknown: manifest unknown
    - This is a registry error. It means there is something wrong with the image on the registry side not your side. Maybe try to pull the same SW from another repo in docker hub 

- is there a common alias for the singularity cmd that everybody agrees upon (i'm lazy ;-))
    - Not commonly in use, but remember that `<tab>` will autocomplete when typing commands, so just writing `sing<tab>` should autocomplete to `singularity`


- I need to visualize my results using Tecplot which is not installed in HPC computers. Can I use container to install Tecplot and visualize my results? They are quite big that can not opened in my local workstation
    - It should be possible to do this, however, note that Tecplot seems to be a licensed software so there might be some licensing restrictions
        - I am using Tecplot in my computer using the license from my department. Can I use the license for HPC as well?
            - You will have to read the license to check for that possibility as we do not have access to your license and what you can do with it


- How does it work to mix Docker containers with Singularity files?
    - Docker containers will have to be converted to Singularity `.sif` images to be usable on our systems
    - The `singularity` command can directly convert docker receipts into Singularity images, [more information can be found here](https://singularity.hpcng.org/user-docs/3.8/singularity_and_docker.html)
        - Thank you very much, really interesting!

- Is it possible to use docker containers on saga/betzy/etc? 
    - No, docker containers needs to be converted to Singularity images, see answer above for more information

- When we download (pull) a Singularity image, is it downloaded into the login node? If it is so, isn't it a problem for this node if the image is too big?
    - When pulling Singularity images they are cached in your home folder which could fill up your quota
        - So, would it be possible to pull into other "worker" node?
            - I would recommend downloading to your project area or at the very least `$USERWORK` and if you do not want it to cache to your home folder use the argument `singularity pull --disable-cache`
            - Another useful command: `singularity cache clean` will remove everything in the cache dir
            - You can also set the `SINGULARITY_CACHEDIR` environment variable to somewhere outside $HOME 


- What is the difference between paths and mounts in singularity?

- How to implement singulaity in slurm file in a correct way? 

- As Metacenter/NIRD administrators, what do you prefer from users, that we ask to install new modules, or that we use containers?
    - It very much depends, if the software is already installed on our system and you just need a newer version we are very much happy to install modules
    - If you expect many people to use the module then contact us and we can install the module
    - If the software is targeted at a specific system (e.g. Ubuntu) or needs packages that are not easy to install then a container is the correct answer
    - **However!** We never ever frown at receiving support tickets from users. We might reply that software X is better suited for containers, but our goal is always to help you perform the research!

- Sabry could you show us how to use singularity pull docer for plasmidfinder (tool). I tried but it was not working. Genomicepidemology second please:) Perfect=it makes sense:) But it works:) Thank you
  - shown live 
  - the command is `singularity pull docker://genomicepidemiology/plasmidfinder`
  
- Where should we pull the singularity to? Which location on Saga?
    - Somewhere you have write access, $HOME, project or work space
    - But note that images are quite large, so it might fill your disk quota on $HOME, if you are close to the 20GiB limit
        - where do I find the work space again?
            - The bash variable `$USERWORK` (use `echo $USERWORK` to see the path) contains the directory of your user workspace
            - Be aware of : NOTE: Current $USERWORK autocleanup period is 42 days. As stated in the login text.


- In actual usage, should the singularity file be moved to work space and fetched back after he job has finished?
    - You store the .sif image in your workspace and copy it to the cluster to run the job, and because you only copied it you don't need to transfer it back again
    - you don't really need to copy because all directories ($HOME, $USERWORK, /cluster/projects) reside on a single parallel filesystem - but be aware of constraints of these different locations (storage quota, retention period ...)
        - what about performance in I/O. 
            - all the same, because it's the same file system (everything under /cluster)
                - really? 
                    - yes, unless ... :)
                    - 
                        - uhm, why are we then using work space? I guess it comes from past habits,or...
                        - because there is no quota on $USERWORK
                        - it's also auto-cleaned, so even if you forget to cleanup the system should do that for you
                            - cool, thanks!

- Could anyone paste the Singularity document link that Sabry uses here?
    - https://documentation.sigma2.no/software/containers.html?highlight=singularity
    - You can also find all the material on top of this hackmd document

- Is there a nice way to install singularity on MacOS, or is there a good workflow to build and run singularity containers on MacOS?
    - https://singularity.hpcng.org/admin-docs/master/installation.html#mac
        - Thanks. But this approach seems rather limited as it only runs from within a vagrantbox, so I was hoping there was another way to run it:-) 
            - Linux container runtimes like SingularityCE cannot run natively on Windows or Mac because of basic incompatibilities with the host kernel. (Contrary to a popular misconception, MacOS does not run on a Linux kernel. It runs on a kernel called Darwin originally forked from BSD.)
            - Thanks. Do you by any chance know a nice way to bind a directory from the host machine to the vagrantbox, and to excecute commands within the vagrantbox on a command line fashin from the host?
                - You may sync directories from the host: https://www.vagrantup.com/docs/synced-folders/basic_usage
                - In general, use Mac to only build the container is a good practice

- Can I get jupyterhub via Docker?
    - https://hub.docker.com/r/jupyterhub/jupyterhub/

- How do you print what the singularity file does?
    - Use `singularity exec <image name>.sif cat /singularity` will print the default execution when run with `singularity run`
    - Or do you mean what the Singularity receipt contains?
        - I guess both.
            - For the second you can read the receipt on the webpage that you want to download from

- I get this error:  could not open image /cluster/projects/nn9987k/sabryr/phreeqc.sif: image format not recognized
    - What command did you run?
    - singularity exec phreeqc.sif cat /singularity
    - that was the only singularity for this software that I could found
    - I had a look and the phreeqc.sif file is not a singularity image, it's a plain text file, html file

- How do you print the version of the singularity file?
    - Use `singularity inspect <name of image>.sif` it will output all the information about the image which can then be further filtered (see `singularity help inspect`)

- how do you remove the image cache in your home directory?
    - https://sylabs.io/guides/3.1/user-guide/cli/singularity_cache_clean.html
    - You can use `singularity cache clean` to remove all cached downloads

----------------------------------
Long thread following (broken formatting):

- How do I know which images I have downloaded previously? (and how to do what's asked in the previous question ;))
    - You can use `singularity cache list` to show the images that are cached
        - only states that I have 4 container files ... There are 4 container file(s) using 871.38 MiB and 35 oci blob file(s) using 863.92 MiB of space Total space used: 1.69 GiB
            - That seems correct, or is there something wrong you think?
                - I'd like to know which images I have downloaded. (not just how many - sorry for being picky)
                    - No problem! Could you try `singularity cache list --verbose`?
                    - sure. a long list starting
```
NAME                     DATE CREATED           SIZE             TYPE
09815c5f37d2d8d87131e1   2021-10-14 14:24:58    51.58 MiB        blob
11bea08770a32e30917346   2021-10-14 14:24:59    6.45 MiB         blob
1b26e7e7420778c281756c   2021-11-04 11:06:25    0.17 KiB         blob
```

- is there any way to figure out to which images these things belong to?
    - It seems the only way is to do `singularity inspect $HOME/.singularity/cache/blob/09815c5f37d2d8d87131e1` where the last part `/blob/<number>` can be extracted from the `singularity cache list --verbose` list
        - partial success: works for lines where type is oci-tmp, not for the blobs ... 
                        
```
singularity inspect $HOME/.singularity/cache/oci-tmp/8f343633898c500138979065b62ecb50be4a29f8e7adfadd8f0b168d2642eab1
org.label-schema.build-arch: amd64
org.label-schema.build-date: Thursday_4_November_2021_11:7:2_EET
org.label-schema.schema-version: 1.0
org.label-schema.usage.singularity.deffile.bootstrap: docker
org.label-schema.usage.singularity.deffile.from: tensorflow/tensorflow:latest
org.label-schema.usage.singularity.version: 3.7.4-1
```
- Is there nothing under `ls $HOME/.singularity/blob`?
    - there is
```
singularity inspect $HOME/.singularity/cache/blob/blobs/sha256/09815c5f37d2d8d87131e14e10a137967abea40bbb74888e4624281dd0a55ab3
FATAL:   Failed to open image $HOME/.singularity/cache/blob/blobs/sha256/09815c5f37d2d8d87131e14e10a137967abea40bbb74888e4624281dd0a55ab3: image format not recognized
```
- Then I'm out of ideas, will ask someone else to follow-up
    - all fine, just need to remember to use the lines with type==oci-tmp only ... that's sufficient, thanks for the help (inspect ... did the trick :-) )
    - please note that the `cache` does not keep track of all the _actual_ images you have downloaded. So even after `singularity cache clean` the image files will still lie around on your file system taking up space, so in that sense you need to keep track of them yourself

- Sabry, can you re-explain the use of `--bind` as when I try `singularity exec --bind /path/containers/data:/data2 bioperl_latest.sif head -n2 /data2/input.txt` as for me it fails to mount
    - Are you literally using "/path/to/containers/data"? Should be the actual path to your data.
    - I had left the _containers_ word in the path! thanks! :)
        - :+1:
    - just to make sure I understand: this means that the files I want singularity to access need to be in the same directory than the container image?
        - No, it needs to be in one of the directories that are _bind mounted_ into the container. On our systems, the default is that your `$HOME` is mounted automatically, which means that you can access anything that is in your home regardless of where the container image is located (it will be called `/cluster/home/$USER` also inside the container). For anything else (`/cluster/project`, `/cluster/work` etc) you'll have to explicitly `--bind` the folder you want to access.
        - And with `--bind` you _can_ give the directory a different name inside the container than what it is on the host, as Sabry was doing above (`--bind <name/of/path/on/host>:<name/of/path/in/container>`)
        - thanks, much clearer! so when Sabry was renaming `/cluster/projects/home/$USER/data:/data2`, the absolute path to the _input.txt_ file _inside_ the container became `/cluster/projects/home/$USER/data2` ?
        - no, the name inside the container becomes simply `/data2/input.txt`
        - ok, understood :+1: 

### Singularity on HPC above this line
---------------------------------
### NIRD Toolkit below this line

- Will issues with setting up ncview fit into todays session? (I wasn't expecting this, but I was wondering since there is a vnc application in the nird toolkit)
    - I think we can have a session on the vnc at the end of the second session
        - Perfect!
        - Update: I actually suddenly figured if out, so unless anyone else is wondering about this, then nevermind :)

- Anybody had a problem logging in to apps.sigma2.no with Feide? I get a "Service not activated" error with my NMBU Feide. The NMBU administrators are working on it but they had little clue what to do or how to activate "Uninett Sigma2 Apps".
    - If the feide administrators does not allow for NIRD toolkit services, the solution is to get a OpenIDP feide account, I think we have sent out some instruction on how to set up this guest feide accounts.j Let us know the eduppn of the guest feide account and we can update your profile on our side.
    - Here is the message that you recieved via email *We also need your correct eduPPN in order to give you acces to the NIRD Toolkit hands-on session. Please see the information below in order to find your eduPPN    
    - In order to use the NIRD Toolkit during the course you will need either a Feide account or a Feide OpenIDP guest account. See your eduPersonPrincipalName (eduPPN) at http://innsyn.feide.no.  You can create a Feide OpenIDP guest account at https://openidp.feide.no/. Those who are using Feide openidp guest account should go to minside.dataporten.no to find eduppn, and it is the field called “Secondary user identifiers* 

- I can signin to apps.sigma2.no but when I try to access the apps I get the message "Service not activated. In order to use this service, a Feide administrator for your organization must activate it. Name of service: jupyterhub"
  - This is a similar problem to the one above, so we suggest feide openIDP.
Users that are able to log in and access the toolkit, but not access services must use their regular Feide when logging in to NIRD toolkit and deploying, but OpenIdP to access services. To make this work add the Craas2,ns9988k group to the authorized groups when you start the service.
    - Please see the issue and workaround [here](https://opslog.sigma2.no/2021/06/25/service-not-activated-in-nird-service-platform/)
- About NIRD. I found out empirically that one cannot submit jobs from the nird folders mounted on fram. Is it so that SLURM does not have access to NIRD folders mounted on fram?
    - NIRD folders should not be directly used from compute nodes. Data on NIRD should be copied to a cluster files system
    - e.g.  /cluster/work/users/$USER 
        - yes, and I guess the point is that the slurm job script is executed by compute nodes, so NIRD in not available there. Therefore, files have to be copied before the job start, and cannot be copied during the job (not even at the end of it).
            - That is correct
                - a bit annoying, though, because it means that the file transfering cannot be automatically linked to job execution. right?
            - Correct. The main reason is slow IO. i.e. the access to data from compute node to NIRD mount is slow and it will be a bottle neck (slow down jobs)
                - ok

- So if I want to use the data stored on Nird on Fram on the compute nodes I have to `rsync`? But if I'm in my working directory then I can use the data from Nird? 
    - More info [on file transfer](https://documentation.sigma2.no/files_storage/file_transfer.html)

- is the NIRD toolkit quota to be considered part of a project quota, or the request to enable the toolkit is essentially an application to extend the project quota?
    - Storage quota in the Toolkit is the same as your NIRD project quota, however compute quota is seperate from this and is connected to a namespace on the service platform. (See the link below to getting started).
        - I meant cpu quota, actually.
    -  You can apply for resources through the application form.
    - For more information, please check out the [getting started guide for the NIRD Toolkit](https://documentation.sigma2.no/nird_toolkit/getting_started_guide.html)
    

- is it possible to use containers (docker/singularity) within the NIRD toolkit, e.g. within the JupyterHub tool?
    - Yes, [you can use a custom image on NIRD](https://documentation.sigma2.no/nird_toolkit/custom-docker-image.html)

- is it possible to also run either singularity or docker from inside a base image, or do we need to add either of these options into our custom image?
    - You should avoid running docker in docker, or singularity in docker
    - why exactly (it would be very convenient for making interfaces with complex workflows readily available on a JupyterHub for instance)?
    - Docker in docker requires mounting some of the host system docker related files and this is a sketchy solution to a problem that should be possible to solve in a cleaner way
        - This is not supported at all
    - Singularity in docker is less sketchy but still not good practice, you can do it but should consider a cleaner way of doing it
        - No one will stop you though
    - Thanks. If possible this would indeed be excellent in order to mimic a complex environment directly on the NIRD toolkit, which could then be used to set up similar enviroments on local machines, the HPC resources, and for (pre- and) postprocessing on the NIRD toolkit
    - If all you need is a development machine, "use and abuse", then [NREC](https://www.nrec.no/) is something to consider too
    - Thanks, but not excacly. We are developing a platform for our users and developers, and believe NIRD toolkit could be an excellent first entry for joint projects, both to set things up, to try it out, and for postprocessing with e.g. ML tools. The plaform is meant to allow for an easy switch between either native builds (ie. ´module load´) or singularity/docker images for various softwares and libraries needed by our developers and users
    - Sounds useful, I'm sure you will find the best way to develop it, either way NREC has worked well for us at UiB so it's a good option if a bare VM is required for something semi permanent
    - Okay, thanks. Will definitely try this option out then


- Does postprocessing happen on the NIRD login containers, and how much memory is available for this?
    - These are with basic resources. The login containers have 8 cpu and 64 GiB memory. 
    - On the nird toolkit You could order large images up t0 160Gb RAM, from the info I got.



- Will we keep access to these NIRD toolkit Resources after the course is over?
    - The resources are available for a week time

- Do you have recommend to preferly choose Jupyter or Jupyterhub on apps.sigma2.no for the user to share their code?
    - From Siri's answer, to share your data Jupyter and Jupyter hub for a group.
 
- A quick question about task number and time. If I ask for more than one task, and 5 minutes, will my job last more than 5 minutes?
    - The job lasts for however long it takes to finish
    - If you e.g request 5 minutes but it finishes in 2 you don't spend the remaining 3 minutes, they are returned to you

- Upon trying to 'install' I get "You are not allowed to use any projectspaces." I guess this is the end of it.
    - Try logout and login again. Might work. If not send a direct message to the host in zoom
        - I will, problem persists.
        - make sure that you accept the dataporten invite which was sent to you yesterday via email
            - accepting invitation solved the problem

- My project has a JupyterHub, I encountered the problem that I have with read/write access for files. To log in to the JupyterHub we had to use an openFeideID. When I have .py files I worked directly on the server, I'm not allowed to edit them on the JupyterHub. Any suggestions on how to overcome this issue? The same is true for the data access.
    - I think that if you make the file group writable, then you can edit in jupyterhub
    - It could also be that the data is shared "read only" this can be checked in the configuration of jupyterhub

- How do I access something someone else has installed? (Sorry of unclear, but I don't really understand the terminology...) 
    - It depends on how it is installed,
    - if it is installed in the image, then it will be accessible,
    - if it is installed in the NIRD project area, you could try running it in a terminal of jupyterhub, but it might as well be run on the nird login node. 
    

- Can we configure the jupyterhub so that all the files in a project space can be accessed and modified through jupyterhub (the whole project space is "shared"?)?
    
    

- I installed it but I cannot access it. I get this error: Service not activated
-- this is controlled by the feide administrator at each university, so we suggest creating a guest feide user, and log into the new service with that account. That account also needs to be in the dataporten group.  
-- Having a similar error. The application is running, but can't get in the URL. Error message: `In order to use this service, a Feide administrator for your organization must activate it. Name of service: rstudio`

- Can we get JupyterHub on the other sigma2 services, like FRAM, SAGA, BETZY?
    - You could start a Jupyter notebook on one of the HPC systems, but not recomeneded. As the HPC systems are to do the processing as jobs (batch processing).
    - A feature request would to be abale to submit jobs from NIRD tool kit, which I think they are working on at the moment.


- a bit of a phylosophical question: while I see the organizing of a course on the toolkit as a very powerfull option, wouldn't this mask the getting of the jupyter notebook up and running with the right environment. Effectively, it would make students less able to do-it-yourself.
    - Correct. So the question would be is the course teaching "jupyter notebooks" or using jupter note book to teach something else (e.g. python). 
    
    

- in the next demo it would be nice to see how we get actual data from NIRD while within one of the services from the toolkit. 
    - The course folder in this example existed on NIRD so as long as you configure it to be shared . Right now we have called the course folder as data instead. So if you configure your hub to have the folder data as the shared folder you will see data from nird in the jupyter hub.    
    - as "persistent storage" right? 
        - Yes. In the above case the persist storage should be selected, and then having data as the shared folder is a default, just remember to check that you want shared storage. Otherwise each user will get a folder on the NIRD project named after their dataporten ID, and they will only see this area of the nird project. This is specific for the jupyterhub.
        - 
    - You can see more info on [persistent storage](https://documentation.sigma2.no/nird_toolkit/persistent-storage.html)
    
   -  all applications store their data locally. This means that iif an application is stopped or restarted without enabling persistent storage, all local data is lost.


- How realistic is it to enable external users? (no Norwegian accounts from university, for example, no Feide)
    - Example: a collaborator from Canada with no officiel ID in NRIS/sigma2/University or similar. I know this person I work with, and I need to give access to data and a jupyter notebook.
    - It is possible to create an openIDP feide guest account from any email address. So as long as your collaborator creates such an account you can give them a dataporten invitation link . 
        - "we only need an email to create an account" guess "we" is NRIS admin, but do USERS have the control or capability to create an account for the external collaborator?
            - From Siri's answer, it seems yes. I would say try it with your own private email (e.g. gmail) and see if it works or is there anything to be corrected or improved-


- Maybe this is actually a question about the difference between Docker and Singularity: Saga supports Singularity instead of Docker, but why NIRD Toolkit uses Docker instead of Singularity?
    - On NIRD you only get access to the running docker container but on Saga you by default get access to the host system. When you have host access as well as docker access it introduces a potential security risk that is not desirable, while on NIRD this security risk is much smaller and not regarded as a problem since you only have access to the container.

- Could you give an example how to load some libray in Jupyter under Toolkid? Like the error when loading: 
```
"import matplotlib.pyplot as plt

from netCDF4 import Dataset
import numpy as np" shows: ---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-5-b0d2d3b01a51> in <module>
      1 import matplotlib.pyplot as plt
----> 2 from netCDF4 import Dataset
      3 import numpy as np
      4 
      5 import tensorflow as tf

ModuleNotFoundError: No module named 'netCDF4'


import tensorflow as tf"
---
```

- How can I have different conda enviroments on Jupyterhub in NIRD toolkit?
    - I think (as I have not tried) this could be activating different kernals, can you see [here](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874) if that is a solution
    - But I would have to define the conda enviroments using the docker files right? I can't use conda within the hub server?
    - Will get back soon on this
- I want to follow up on the above question. Can each member of the JupyterHub create their own python environment?
    - Will get back soon on this
        - Could you write something here on how to do it?


- I guess each session of work in the Toolkit is boud to the browser session. Is it so that a blicnk in the network will cause loss of data? Is it possible to launch non-interactive sessions? Or does the server survives anyway so once launched it is just there available?
    - The server/app/container stays up independent of internet access or browser sessions
    - So if you lose or close the connection nothing will disappear

- Sorry if this is not so relevant to todays subject. Why am I not allowed to choose `--ntasks-per-node=16` `--cpus-per-task=16`, (16*16=256) on Betzy when there are 256 (virtual) cores to choose from?
    - On Betzy you are supposed to use the whole node. So using `--cpus-per-task` is enough  
        - Follow-up q1: Choosing `--tasks-per-node=8` `--cpus-per-task=16` works fine, and from `htop` on a node I see that 64 cores are used from each CPU. Meaning no SMT is in use. I want to try to use SMT, so i tried increasing ~~either of the mentioned parameters~~ `--tasks-per-node` to 16 ~~while leaving the other at 8~. Then I get the error `sbatch: error: CPU count per node can not be satisfied` `sbatch: error: Batch job submission failed: Requested node configuration is not available` when submitting the job. Why cant I request 256 (virtual) cores per node?
            - Can you please share the exact resource request when this happens (the SBTACH request)
                - `sbatch V50_gxpf1a.sh` is this what you need?
                    - no the #SBATCH lines inside V50_gxpf1a.sh
                        - `#SBATCH --nodes=4 #SBATCH --ntasks-per-node=16 #SBATCH --cpus-per-task=16` along with job and account name.
                        - I will try to investigate and come back to you
                            - Thanks =)
                        - CPU cores per node 128 not 256 on betzy
                        - [sabryr@login-1.BETZY ~]$ sinfo -n                               b1113  -o "%c %n" 
                            CPUS HOSTNAMES
                            128 b1113
                            - yes, but there are 256 HyperThreads. So you can actually run 256 MPI ranks, or 256 OpenMP threads. Performance will most likely be worse for all practical purposes, but it is technically possible.
                                - Yes, I heard from you guys that using SMT is probably not a good idea, but the author of the code I'm using says that he has had a small performance improvement when using SMT. Well, he was testing on an Intel system, so hyperthreading there, but the principle is the same?
                                    - sure, the principle is the same and it is of course worth a try. 
        - Follow-up q2: [The documentation](https://documentation.sigma2.no/jobs/parallel-calculations.html) says that a hybrid solution yields the best performance, stating `--ntasks-per-node=8` `--cpus-per-task=16` specifically. Should I not do that?
            - What I meant was ntasks-per-node not needed to achive what you are intending.
        - It might work with `--ntasks-per-node=16 --cpus-per-task=8` but then override with `export OMP_NUM_THREADS=16` in the run script
            - So Slurm is not allowed to oversubscribe the nodes, but you are allowed to override once you have gotten the resources.
                - Aha, so I request 16 MPI ranks with 8 OpenMP threads each, then I override `OMP_NUM_THREADS` to 16?
                    - `OMP_NUM_THREADS` is set by SLURM, but it is OK to modify it
                    - yes, that is perfectly fine
                        - I'll just try it now and see what happens!
                    - it's also _possible_ to oversubscribe the MPI ranks, with OpenMPI: `mpirun --np <2*SLURM_TASKS> --oversubscribe ./myprogram`. Whether it's a good idea is of course a different story :slightly_smiling_face: 
                        - Thanks =) `SLURM_TASKS`, what parameter is that?
                        - Probably `$SLURM_NTASKS`. Should be the env parameter set by Slurm representing the total number of tasks
                            - Allright, I'll test and see what happens! Thanks!
                        - You'll probably have to be very specific with task placement in the `mpirun` command for all of this to work. Options like `--map-by`, `--bind-to` etc
                            - Hmm... Sounds complicated!
                            - oh, but it is :slightly_smiling_face: but `htop` is your friend here
                                - Agreed! :wink:

- I guess since the GPUs are already taken we are not supposed to run the installation of the deep-learning service, right?
    - there is no gpu reservation for this course. so we recommend to use cpu
        -oh, right they were not taken: they were not there in the first place :D

- when I try to install Jupyter on _https://apps.sigma2.no/_, I get an error message: _Error installing package failed to register client, dataporten returned status text: '403 Forbidden' and body '{ "message": "Insufficient permissions" }'_ . I did accept the invite for the _CraaS2, ns9988k_ group... I guess I don't really understand how to use the NIRD toolkit...
    - and when I try to acces the application through the `My applications` tab, I get `Not authorized to access resource`
    - which machine type did you select?
    - ok, so I _can_ access the jupyterhub _craas3-ns9988k-hub_ created by Siri. But I cannot create my own jupyter notebook, even when selecting _Base_ in Machine type: error _failed to register client, dataporten returned status text: '403 Forbidden' and body '{ "message": "Insufficient permissions" }'_
    - same problem with the rstudio and vnc installations

- How fast is the file system on NIRD compared to those used in the regular HPC systems?
    - As fast as the cluster partiions on HPC systems. But when accessing from NRID toolkit tools. Not from HPC systems. 


- Can we (users) create packages to be run on the toolkit? Beyond the 7 that are there already.
    - Yes, but need to ask the admins
        - well, "aks admin" means "no, users cannot"

- To finalize my understanding: The Nird toolkit is a IAAS, effectively, as is NREC, with the difference that the toolkit is by definition linked to the NIRD storage AND provides a few packages that can easily be installed right away, while NREC provides flavours of OSs on which we can create what we want.
    - thanks for trying to clarify. what is _NREC_?
        - https://www.nrec.no/ Sorry, I gave it for granted since it was mentioned earlier.

- If I want to create a JupyterHub on a project, do I need to be the project owner or can I just create a JH for my group as long as I'm part of the project?
--The project leader creates the dataporten group and asks for NIRD toolkit to be enabled. After that you can create the JH when you have accepted the invitation to the dataporten group.  

- I'm not getting the "Data" (the link to NIRD persistent storage) in the Desktop VNC service.
    - Found it: it's under /mnt/craas3-ns9988k


## Feedback for the day

- One thing you particularly enjoyed
    - I finally know what the NIRD toolkit is about. Thanks!
    - I think I got a better idea on how to get a JupyterHub via dataporten.
    - I run Firefox visiting the NIRD toolkit, which was running the Ubuntu desktor, which I used to run Firefox. Recursion is always cool!
    - pre-recorded videos can be useful for after the course, especially if they are split into indivdiual tasks (but not during a zoom session)
    - Nice to see how to get your hands on and launch a container.
    
- One thing we should change/improve for next time:
    - I guess I got used to collaborative teaching (code-refinery style), so I would suggest to go for collabortive teaching live, rather than streaming videos. Much better human interaction. :+1: 
    - I'm still not quite sure on how to use containers and how to use them. 
    - more time on containers and how to use them to execute applications. More practicetime.
    - please no pre-recorded videos :) :+1::+1: 
    - NIRD toolkit: I am still not very clear of what this is about, and what advantages it has over just using the applications by themselves on a cluster
    - The part on toolkits examples should be more focusses on what the toolkit is and how and why to configure it, rather than the actual content of the example (in particular, one we get the jupiter notebook running who cares what the accuracy of the shoe prediction model is. The point is to get it running). :+1: 
    - If possible, please, send the course info 24 h before the course starts so we have enough time to check that we have access to the right resources and be able to get any problems fixed in good time. (All your kind help was very much appreciated to sort out issues though!)
    - If you want students to work hands-on during the session, then provide time for it. I find hands-on to be very good for learning, but personally, I cannot follow a talk/demo and perform it myself at the same time. I lose track and therefore decide to drop the hands-on activity. If time does not allow us to try everything out ourselves, then perhaps you can briefly introduce the knowledge/commands and give us a task where you let us try out just the basics. Briefly show the solution afterwards. Save the more advanced stuff (probably without hands-on) until the basics have been settled.
 
:::warning
Please write always above this ^ line. If you do not need to edit this hackMD, please switch to  "View" mode (top, icon that looks like an eye :eye:)
:::