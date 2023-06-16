---
orphan: true
---

(training-2023-spring-best-practices-notes-day3)=

# Questions and answers from Best Practices on NRIS Clusters May 2023, Day 3 (May 11)

```{contents} Table of Contents
```

## Icebreaker questions

- Which software would you like to run on our HPC machines?
    - COMSOL and Cogenda Genius
    - I hope it is relevant, but I am curious about how to open jupyter notebook, when we are not project leaders.


- When you need/install new software, is it typically ...
  - Python: ooo
  - Conda:
  - R:
  - EasyBuild:
  - Container:
  - C/C++:
  - Fortran:
  - Not sure: 

- Do you want to learn about GPU programming ?
Check this next event: https://enccs.se/events/list/?tribe-bar-search=gpu+programming

- Which GPU programming model do you use?
  - OpenACC:
  - OpenMP offloading
  - SYCL:
  - CUDA:o
  - HIP:
  - OpenCL:
  
- Which ML framework do you use?
  - PyTorch:o
  - TensorFlow:
  - Jax:
  - Others:  

## Questions and Comments 

### Installing software
https://documentation.sigma2.no/software/modulescheme.html

1. How to unload a module?
    * `module unload [module]` Removes just that module
    * `module purge` Removes all loaded modules

2. How about mamba? Worthy to make mamba available along with conda?
   - good suggestion! mamba is an alternative implementation to conda which offers sometimes faster dependency resolution
   - RB: personally I use micromamba in a container: https://github.com/bast/singularity-conda

3. How to do "conda init" without modifying .bashrc
    - `source $EBROOTMINICONDA3/bin/activate `
    - how is it for anaconda?
      - `source $EBROOTANACONDA3/bin/activate`

4. How about this path
 Source the conda environment setup
 The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}

```
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
```
    - is this in the documentation? is this a question or an advice? :-)
    
    Question?
    Source: https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/installing-software-python.md
     - see above: `source ${EBROOTANACONDA3}/bin/activate` (or replace by miniconda)
     - it could be that the uio suggestion also works

5. How about if we want to use a jupyter notebook that other person created, when we are working at the same project?
   - is the notebook then saved on the cluster or where is it saved/shared? 
   -In the cluster/projects/
      - probably NIRD toolkit is a good solution for this

6. Suggestion on how to run jupyter notebook in compute nodes having GPU as GPU is not available in login nodes?
   - probably NIRD toolkit is a good solution for this

7. Hm I remember that they told me to use NIRD Toolkit the project leader should do this and give afterwards permission to other users. Might i misunderstand.
    - the project leader should apply for NIRD toolkit to be enabled on a NIRD project to sigma2, and add the people that should be allowed to start a jupyter notebook to a feide group that will be connected to the project. This is a one time thing. After that the project leader does not need to do anything else.

8. This ticket may be relevant for this discussion
     - The user has wanted to shorten the long custom path
     
    - https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment
        - This explains how you can set the env_promt to whatever you want (this case, remove absolute path)
        

9. This question may not be relevant: if I need HPC access for 3 years project, how should I apply for the access. Should I apply for SSEW and renew it many times?
    - you can apply for a regular HPC project here: https://www.sigma2.no/high-performance-computing . The application needs to be renewed two times a year. The allocation period is half a year at the time. 
    - adding to this: the renewal is typically relatively easy, you don't need to re-write everything every time. the motivation why quota is allocated on a 6-month basis and not for next 3 years is that often applications are not very good at estimating what they will need (some need a lot more, some use only 10% of what they ask for) and also this gives more flexibility to adapt to new hardware procurements.
    - all HPC questions are relevant and welcome :-)


### GPUs


Our general contact email is  support@nris.no

