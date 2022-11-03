---
orphan: true
---

(training-2022-autumn-best-practices-notes-day3)=
# Questions and answers from Best Practices on NRIS Clusters Nov 2022, Day 3 (Nov 3)

```{contents} Table of Contents
```

## Icebreaker Question 
- Which software would you like to run on our HPC machines?
    - genomics/transcriptomics tools: assemblers (Flye, CANU, Trinity), alignment (bowtie, BWA), polishing (Racon, Pilon), chromosome assembly (Ref Aligner, Ragout2), filtering, decontamination (Kraken, Tiara), quality check (FastQC, BUSCO, Quast), mapping (BLAST, DIAMOND)
    - amplicon analysis pipelines (Qiime, anvio)
    - gene calling (Braker2)
    - Visual Studio Code
    - Python code installed with Conda
    - Containers
    - Nextflow pipelines
    - My own python application


## Questions and Comments

### [Installing software with EasyBuild](https://documentation.sigma2.no/software/userinstallsw/easybuild.html)

1. How can we install globally with easybuild? Locally just fills up the project folder.
    - Request it from us. If many users need it can/will be installed globally. 
    - but please also note that while it creates many temporary files, the actual binaries/libraries (install target) should not fill the quota. in other words, I don't think you need to keep all files downloaded/generated during an installation. I would be surprised if the actual binary/library fills the project quota. binaries/libraries are on the order of very few files and on the order of megabytes typically.

2. [Maiken] Easybuild also ensures that you can have multiple versions of the same software without causing conflicts. And makes sure all dependencies are included with the right versions. 

3. What is Lmod?
    - https://lmod.readthedocs.io/en/latest/ - it is a system that allows loading software as modules so that the software is available in its own environment. You can load software and unload (purge) it. We have shown examples of this earlier in the course. Issuing the command 'module' will print a few hints and some info. 
    - is it the program behind "module load"? - yes.
    - Lmod is written in Lua (a small easy to learn language from Brazil, www.lua.org, it's also the scripting language used in Minecraft).
      - yes very popular in game development since it's so small and can be shipped together with the game

4. "foss": free open source software

5. Documentation suggests searching with `ls $EBROOTEASYBUILD/easybuild/easyconfigs/` - not with `eb -S`, is there a difference?
   - we will adapt documentation. both do the same thing but we now recommend `eb -S`

6. I got an error using easybuild: FAILED: Installation ended unsuccessfully (build directory: /cluster/home/gyorkeia/.local/easybuild/build/MiXCR/3.0.13/system-system-Java-1.8): build failed (first 300 chars): Checksum verification for /cluster/home/gyorkeia/.local/easybuild/sources/m/MiXCR/mixcr-3.0.13.zip using f0b32efadf6dae95819cd90cf49978e48797fc9f2685af1fd282a47d3f9fda08 failed. (took 0 secs) What does this mean?
   - to me it looks like incorrect configuration. in the configuration dependencies are verified with checksums to make sure they are authentic and here a dependency has changed but the configuration hasn't? (but I know too little about easybuild)
   - checksum: unique identifier/result which will change completely if the thing that is checksummed changes ever so slightly
   - typically this is because the checksum written in the easyconfig file and that of the source are not matching. 
   - FYI: the checksum used is `sha256sum`
              
            
7. Would it be safe to just always install with `--robot`? This is what documentation previously suggested.
   - yes, but verify first with dry run (this is what I gathered from the discussion)

8. What is sanity checking?
   - making sure that it looks alright but I was distracted and not sure in which context precisely it was mentioned
   - This are tests to make sure the software was installed properly
   - If you know what pytests are , this is something like that.
   - Some operations are perfomed on that has a known result (or an acceptable range), then the output is compared with the known values (or fals within the eaccepted range)
   - Sometimes it is just checking the expected files and directories are present after instlation is complete. 
    

9. Where should I install software home or project area?
    - Project folder is recommended.
        - Installing into the project folder also allows you to share with colleagues as others can also use `module use PATH` to access the software
    - Home folder could be convient though, so if it is a smaller installation (size and number of files) then install it in HOME
 
 
 
### [Software Installation - Python Packages](https://documentation.sigma2.no/software/userinstallsw/python.html)

(but it's still fine to ask more questions about EasyBuild)

10. What is  virtual environment?
    - an isolated environment where Python dependencies get installed into, instead of installing them system-wide or into the module system
    - it's a folder and everything we install then when the virtual environment is activated, will get installed into that folder
    - it's a good idea to have one virtual environment per project
    - advantage of virtual environments:
      - different projects often have different version dependencies and this way this is no problem
      - less scary to install things since if I mess up, I can delete the virtual environment and that's it 
      - in combination with `requirements.txt` (which documents dependencies), more reproducible than system installs since others can (re)create their own virtual environment from `requirements.txt`
    - maybe a good analogy is: my own local disposable environment for Python packages

11. 100 projects, 100 virtual environments, 100 python installations? Space wastage? Is there a way to use a common python package and have multiple virtual envs for project specific modules(medical image processing packages, deep learning packages, statistic, etc)?
    - if they are very similar, it could make sense to reuse them across projects
        - How to reuse?
           - instead of having one per project, you could have one called "deep-learning" and then activate it ("source" it) from various projects in their job scripts 
    - but I still prefer to have one per project
    - by space waste you mean disk quota on cluster? 
        - My local computer taking 706MB for one env folder and I want to reuse some modules across projects to save space.
    - If you have a good setup that fits most of your projects, but need some slight changes per project, you can make a new environment that can access the packages from your main environment. [Found a link](https://stackoverflow.com/questions/10538675/can-a-virtualenv-inherit-from-another)
        - nice!! thumbsup!:)

12. To use a conda package in a jobscript, you need to `conda activate` it. But to activate you need to do `conda init`. But for that it tells me to restart the terminal. How can I implement that in  jobscript?
      - we have an example using a conda environment in a job script : https://documentation.sigma2.no/software/userinstallsw/python.html?#in-batch-job-scripts
          - does not work for me, because it wants me to `conda init` and restart terminal
             - OK let me test it, too (working on it ...)
                 - I might just have messed up that `source` step that is mentioned in the question further down - I think I'll just have to fix that
                 - I now ran it successfully but indeed I had to correct the source step also. We will fix it in the docs. sorry for the confusion
      - and you have module loaded Anaconda/Miniconda? yes

13. Mamba is supposed to be a faster re-implementation of conda for solving dependencies, have you used that?
    - mamba can offer a faster dependency resolution compared to Anaconda but we currently don't have it installed
    - Thanks :)

14. In the conda jobscript you showed, there was no `conda init bash` which you said is needed to run conda commands. Is it not needed in a jobscript?
       -  There is a *source* step in the jobscript that will do this `conda init bash` step in a better way.
     

### [R packages](https://documentation.sigma2.no/software/userinstallsw/R.html)

(ok to still keep asking about Python also)

15. What are your suggestions for version control of packages in R? Anything that actually works well?
    - I have used https://rstudio.github.io/renv/articles/renv.html but I use R only every few months so I am not a "power user"

16. `watch squeue --me` The `watch` command will recall the command every ~2 seconds and update the display to show the latest information
    - THANK YOU! after a decade of typing `squeue -u myself` and writing clumsy documentation: `squeue -u <yourself>` I can finally use and document `squeue --me`
        -`squeue --me -i 5` would call the squeue every 5 seconds if you perfer that. Works with most commands. 
            - Didn't know about `squeue -i 5` option, I always used `watch`. But got to know.
        

### [Singularity on HPC](https://documentation.sigma2.no/software/containers.html)


17. I have sometimes had problems with importing some docker container to singularity, when the program in the docker is dependent on an independent file like a local database. In the conversion the path to the file gets scrambled or something. Any experience with this?
    - Sorry this was a very hard question to formulate.
      - no worries, we will raise the question.
    - I haven't seen this before. If you have an example we can look. But also OK to not share here if you prefer sharing directly to our support: support@nris.no
    - Sorry, no example this was a long time ago. Let me see if I can find an example online.


18. Quick poll: have you used containers?

- I have used Docker containers: oo
- I have used Singularity containers: oo
- (some other containers, please add): o
    - WSL 1 is a container, right? xD
      - yeah it kind of is :-) 
- I have created container images (Docker or Singularity or other): oo
- I have not used any containers yet: ooo


19. Any container technology that you would really like to have on our clusters which we don't have yet?
    - apptainer ("a sister/cousin of singularity"): we know that people want this but we are waiting until it becomes available as package on CentOS (or if we get too impatient, we will build it at some point from sources)

20. Does it bind recursively? What if you want to bind several directories?
    - I have in past specified all separately
    - comma separated list
        - [doc](https://docs.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html)
    - Great!



21. If you are unsure if your problem is worth a ticket/issue/email: it is worth a ticket!



### Feedback for today the day

- What did you particularly like?
  - Informative
  - ...

- What should we improve/change/remove?
  - More interactive stuff oo
  - More feedback
  - add enough session for questions

