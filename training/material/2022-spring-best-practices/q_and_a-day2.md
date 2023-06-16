---
orphan: true
---

(training-2022-spring-best-practices-notes-day2)=
# Questions and answers from Best Practices on NRIS Clusters May 2022, Day 2 (May 24)

```{contents} Table of Contents
```

## Icebreaker question

Which software would you like to run on our HPC machines?
- CP2k, gromacs, openFoam
- CNCI, PfamScan, CNIT, FastQC,
- ROOT
- R
- MrBayes, BEAST, RaXml, HybPiper, HyPhyloMaker etc
- Quantum Espresso Vasp use RUST
- Containers
- Suite2p, Caiman (python packeges for cell detection in >100GB microscopy videos)


## Having a look at available software

[Software module scheme](https://documentation.sigma2.no/software/modulescheme.html)

1. So when we try `python --version` why cant it show the most updated version python 3.6.8 instead of python 2.7.5
    - python and python3 are installations which come with the system but we can bring other versions "into view" using module load. the default packages (python and python3) are not the newest ones because they have to be conservative because many system software tools depend on them. this is also the reason why on this operating system python resolves to Python 2, whereas python3 resolves to Python 3: to not break system software that might depend on these.
    - in your own run scripts it is almost always better to load Python via module load and not to rely on the system default python (in this case `python` or `python3`). reason: the default python version might change but if you actively module load a well defined version, you are sure that it won't change. this is better for reproducibility of your scripts.


## Installing software

[Documentation/material](https://documentation.sigma2.no/software/userinstallsw.html)

2. Where is the optimal location to install software? Login-node, project-directory, home-directory?
    - depends, let me expand:
      - where to perform the installation depends whether it takes seconds (then login node is fine) or many minutes or hours (then via slurm script on a compute node or interactive compute node). personally I prefer using a Slurm script because then it gets automatically documented and I won't forget how I did that and as a bonus I don't occupy the login node
      - if it is just for me and only few files, then HOME is fine
      - often project directory is better because you have more quota (example: Conda installation) and your project colleagues might want to use the installation too and don't have to go through it individually
      - then depends on what language and framework (EasyBuild, Python, R, Julia, Rust, C, Fortran ...), it might be good to install into a Conda environment or a virtual environment or renv or an explicit path location
      - long story short: for most purposes login node and project directory is a good start

3. Which HISAT2 would you install?
    - you mean which version?
    - we have HISAT2/2.1.0-foss-2018b already installed on saga

4. When building with EasyBuild there is a lot of files in the .local folder after the build. After building a few codes like this we come into conflict with the disk limit on the number of files. Is there some way of cleaning up some of this stuff that is not needed after build?
    - in one of the last courses the instructors described a solution to this by installing an environment / packages into a Singularity file to work around maximum file limit (just mentioning because it wasn't mentioned yet)
      - but this was more of a workaround for conda installs
    - there is an option/command for this but unfortunately i am not an EB expert but I will raise it
    - You can install also into your project folder as described  [here](https://documentation.sigma2.no/software/userinstallsw/easybuild.html#installing-software-in-a-project-folder) and then just delete the `.local/easybuild` folder
    - we will also discuss this in the Q&A session
    - How to be more specific when installing.
        -  `eb --show-config` shows where things are placed, specifying `sourcepath` you could place the downloaded files in a specific folder and delete it later [Details](https://docs.easybuild.io/en/latest/Configuration.html) . Removing the  `sourcepath` folder recursively will clear all downloads and would not affect the instaltion.

5. I don't know if I understood correctly, but can I build my own module with EasyBuild for any bioinformatic software?
    - yes, provided that the software is available in the EasyBuild ecosystem (people and communities can contribute installation recipes that others can use)
    - related question would be: how can I find out whether a software can be installed via EB. asking below

6. How can I find out whether a software can be installed via EasyBuild?
    - this is how I check but I am not sure this is the best way: I look for the software here: <https://github.com/easybuilders/easybuild-easyconfigs/tree/develop/easybuild/easyconfigs>

    - maybe with the command : `eb --list-software` yet it depends how the eb i configured
        - produces a lot of output better use `-S SOFTWARE`
    - you can search for available software with `eb --search SOFTWARE` or `eb -S SOFTWARE`, then you only need the filename to install. For example `eb Julia-1.7.2-linux-x86_64.eb`
    - it is always good to check if the software is already available with `module avail software`

7. Related to question 5 and 6: What do I do if the needed software is not available with EB? Can I use `wget` to install the software into my project directory?
    - if the exact version does not exist but another version, you or somebody could contribute to EasyBuild a new recipe but this might be out of reach
    - if it is a binary, sometimes downloading a binary with `wget` might be enough
    - for Python/R software there are other package management tools that can be used (e.g. conda or pip)
    - "traditional" Unix/Linux tools can often be installed by downloading the zip/tar archive, extracting it, followed by `configure --prefix=/where/to/place/it`, `make` and `make install`
    - it can range from easy to non-trivial: do not hesitate to contact us for assistance
    - we will talk about conda in few minutes which can be used for a lot of software packages

8. How do I get cowsay on Saga? :D
    - is it available in EasyBuild? (sorry just kidding)
        - Yes it is. `eb cowsay-3.04.eb`
        - Cowsay is a simple perl script /usr/bin/cowsay (if installed by system as rpm).

9. The software STRINGTIE is not in eb, how to instsall it?
    - good question. see also question 7.
    - do you have a link/website to the software?
    - it seems to be in EB: <https://github.com/easybuilders/easybuild-easyconfigs/tree/develop/easybuild/easyconfigs/s/StringTie>
    - `StringTie/1.3.5-GCCcore-8.2.0` is available on saga


## Installing R packages

[Documentation/material](https://documentation.sigma2.no/software/userinstallsw/R.html)

10.  It would be nice if we had access to the terminal commands as they are being entered so that we could scroll up and see previous commands in order to follow along with the lecture.
     - thank you for the comment. I fully agree that it would be useful and important to show past commands.
     - You could consider including command history after the lecture is over along with the notes from the lecture.
       - good suggestion. we could copy the output from `history` to this place here
     - instructor is also more or less following documentation linked above. but this does not invalidate this good suggestion to show the commands


## Installing Python packages

[Documentation/material](https://documentation.sigma2.no/software/userinstallsw/python.html)

[Working with Conda](https://documentation.sigma2.no/software/userinstallsw/python.html#anaconda-miniconda-conda)

11. I have been always wondering what the channels means in "conda config --add channels XXX"?
    - channels are like collections or distributions
    - conda-forge is perhaps the most popular one (community-driven and curated channel/collection of very many packages)
    - also bioconda is a good example
    - you or a research group or community can create your own channels also
    - if you look for a package in the conda archive, there usually is also information about the channel it is available in

12. May I ask why need to do `source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh` before "conda activate ENVIRONMENT", I thought "conda activate ENVIRONMENT" is already enough?
    - because the shell is not set for conda and you need to show conda which shell it shall be using
        - but why does the module not do this (the sourcing) for us?
            -  I guess because there is a difference between installaiton and initialisation of conda, and the latter is not done by the installaiton procedure automatically. Also the installer does not know which shell you will prefer for conda, it may be different from the one you are using.
                - but we insist on using bash (on the clusters; I agree that on laptops people have various shells) and in which scenario would we want to module load conda and not source the environment? to me it looks like it should be part of the module.
                - maybe the module does not want to assume that we are on a cluster and also work on the laptop.

13. I have struggled with version conflicts between installations in the module and in the venv. When installing a missing Python package, through dependencies it may want to install a different, say Pandas, than what is already loaded through the module. The answer for me has been to install less through modules and more in my own venv, but that feels like a waste. Any other approaches to this?
       - I understand that mixing modules and venv or conda can be tricky. Personally I often try to take as little Python as possible from modules and keep everything in venv or Conda. But there is a risk to this to not use the tuned/optimized modules. Sometimes/often it does not matter. If my Python analysis takes 2 seconds it does not matter that it could be 50% faster and then I prefer venv/Conda. But if you spend a lot of time in the Python code, it may make sense to carefully check that optimized libraries are used.

14. I have 100K limit in my /home.  Virtual environments can get a lot of files. When I create venvs in the USERWORK, I may lose them and therefore tar copy them to one file at the /home, and extract back when the env is needed. Any better way to do this?
    - You can also create your venv environments in your project folder where you have a much higher file quota.
    - alternative (but using project folder is the better answer, but just for info): since your environment is ideally documented in requirements.txt (pip) or environment.yml (conda), you could create the venv also "on the fly" from the file once you need it in your run script since it can be regenerated at any moment. this may take few seconds or minutes though so it would not make sense if the computation is short. I always create venv or conda env from a file, not by installing "directly": this way I can always delete the environment and always recreate it without remembering what precisely I had to do.
    - yet another advanced alternative is to install into a container but I would only do that if I also want to move the container to a different cluster
      - Thank you. I did not consider the project folder. Must check that. Yes, I try to stay away of containers as long as I can. :)
        - good plan. they are great but they also are another layer of complexity so it's good plan to keep things simple
    - You could direct the conda cache to be in project folder or in a temporary location like /cluser/work/users/$USER. You could place the following in .condarc file in your home directory
    ```
     pkgs_dirs:
      - /cluster/work/users/<my_user_name>/cache
    ```
    - Copy that. Thank you. I have given up on Conda due to these issues, but with the help from here, I will give it a try again


## GPU computing

[Previouse course video on GPU](https://www.youtube.com/watch?v=cPKajela6LE&list=PLoR6m-sar9Ai3TMU96xAGDx-UImMzLXae&index=10)

[Documentation/material](https://documentation.sigma2.no/getting_started/tutorials/gpu.html)

- Simple approach is also possible, with a bit lower performance than CUDA which gives the best performance.
    - [Simple offloading in high level languages](https://documentation.sigma2.no/code_development/offloading.html)
    - [Guides](https://documentation.sigma2.no/code_development/guides.html)
    - [A simple Fortran 2008 approch](https://documentation.sigma2.no/code_development/guides/offloading-using-fortran.html#offload-fortran-concurrent)

15. So, GPUs are good at doing linear algebra, but we already have very optimized linear algebra routines in BLAS/LAPACK, can we use BLAS/LAPACK on the GPUs?
     - yes. there is for instance cuBLAS, cuSOLVER, ... so this is possible and often a good and sometimes "easy" (without programming everything from scratch) way to offload work to the GPU
        - last time I tried this is few years ago and there it worked well for large matrices and not so well for small matrices but things have evolved a lot since
     - If you're lucky you can get 10x maybe 100x performance compared to CPU.
     - all the hardware vendors provide solutions/libraries to run BLAS/LAPACK on GPU (different names, same idea)

16. I am a bit confused about the GPU concept. I thought a node contained some CPUs and some GPUs.
     - this is also my understanding (not all the nodes but some nodes)
     - There is a difference between the VGA card (the connection to the monitor; this one is in our laptops but often not present on the cluster) and a GPU used for computation that we mentioned (V100, A100).

17. How do you program an AMD GPU, is it still CUDA?
    - No, it's called 'ROCm' but is very similar to CUDA. In many instances it can actually be automatically translated from CUDA to ROCm using a tool called HIP.
    - More on documentation [Offloading to GPUs](https://documentation.sigma2.no/code_development/offloading.html)
    - {ref}`dev-guides_gpu`

18. How do I sign up to start using Saga / Betzy in my lab at NTNU?
    -  start here : <https://www.sigma2.no/high-performance-computing>
    -  <https://www.sigma2.no/apply-e-infrastructure-resources>


## Extended support

[Documentation](https://documentation.sigma2.no/getting_help/extended_support.html)


## General Q&A

19. How to find (python) packages in a module that may be part of a module but is not in the name?
    - if the package is installed by conda, `conda list`
    - if the package is installed by pip, `python -m pip list`
    - I don't think the answers above answer the question. I understand the question as: How to find out which packages come into view when module loading specific modules. I don't have a good answer to this question.

20. virtualenv vs conda what pro/cons?
    - venv does not resolve the dependencies, it is just encapsulaitng your setup
       - maybe the question was pip/PyPI vs conda
    - pip: all packages have python interface
       - for purely python projects, pip may be enough. venv is more lightweight than conda environment and I often start with pip and in a Python project if I don't find it there, I move to conda
    - conda: not only for python interface packages but more general: can package codes in many languages
    - Conda distribute precompiled binaries (what this means is that someone should have compiled the software for your system , operating system, architecture. PIP on the otherhand has the possibility to compile it from source

22. also, mamba is used in some python package installation instructions, how does that relates to conda/virtualenv?
    - mamba is an alternative to conda and can be used interchangeably with conda. it offers faster dependency resolution but can access all packages that conda can access also
    - mamba and virtualenv are different things but same idea but they cannot be used interchangeably and typically access different package repositories (although conda/mamba can install also packages from PyPI)


## Feedback Day 2

One thing you thought was good:

- the "Software Installation - (SR & JD)" lecture was very helpful


One thing that should be improved/changed until next time:

- A way to scroll through the terminal input during lecture would be very helpful to see previous commands while the lecturer is talking.
- Also including a full output of the terminal history after the lecture with the notes.
