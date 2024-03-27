(nessi-eessi-using)=

# Using software provided by NESSI and EESSI

Once the environment is configured to
[access software provided by NESSI or EESSI](nessi-eessi-access-on-nris), all a
user has to do is loading module, for example,

    module load GROMACS/2023.1-foss-2022a

**Example job script using NESSI software**

We demonstrate the use of NESSI in a job script using GROMACS as an example. Below is a simple job script. 

    #!/bin/bash
    #SBATCH --account=REPLACE_WITH_YOUR_NN****K
    #SBATCH --job-name=NESSI_GROMACS
    #SBATCH --partition=normal
    #SBATCH --mem=16G
    #SBATCH --nodes=1
    #SBATCH --tasks-per-node=4
    #SBATCH --time=30 # short runtime

    module load NESSI/2023.06
    module load GROMACS/2023.1-foss-2022a

    # prepare test data if not yet present
    if [ ! -f GROMACS_TestCaseA.tar.gz ]; then
      curl -OL https://repository.prace-ri.eu/ueabs/GROMACS/1.2/GROMACS_TestCaseA.tar.gz
    fi
    if [ ! -f ion_channel.tpr ]; then
      tar xfz GROMACS_TestCaseA.tar.gz
    fi

    rm -f ener.edr nessi_logfile.log

    # run GROMACS, we use 'time' to measure how long it runs
    # note: downscaled to just 1k steps (full run is 10k steps)
    time gmx mdrun -s ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 1000 -g nessi_logfile

Change the account (in line `#SBATCH --account=REPLACE...`) in the job script
above and store it in a file named `nessi_gromacs.sh`
Then, submit it with

    sbatch nessi_gromacs.sh

When the job has ended it should have produced a file `slurm-JOBID.out` where
`JOBID` is the id of the job that was run.


**Example job script using locally installed software**

We demonstrate the minimal change needed to the run the same software
(`GROMACS`) using the installations provided locally on the HPC systems the
following script just comments out the `module load NESSI/2023.06` line:

    #!/bin/bash
    #SBATCH --account=REPLACE_WITH_YOUR_NN****K
    #SBATCH --job-name=LOCAL_GROMACS
    #SBATCH --partition=normal
    #SBATCH --mem=16G
    #SBATCH --nodes=1
    #SBATCH --tasks-per-node=4
    #SBATCH --time=30 # short runtime

    # module load NESSI/2023.06
    module load GROMACS/2023.1-foss-2022a

    # prepare test data if not yet present
    if [ ! -f GROMACS_TestCaseA.tar.gz ]; then
      curl -OL https://repository.prace-ri.eu/ueabs/GROMACS/1.2/GROMACS_TestCaseA.tar.gz
    fi
    if [ ! -f ion_channel.tpr ]; then
      tar xfz GROMACS_TestCaseA.tar.gz
    fi

    rm -f ener.edr nessi_logfile.log

    # run GROMACS, we use 'time' to measure how long it runs
    # note: downscaled to just 1k steps (full run is 10k steps)
    time gmx mdrun -s ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 1000 -g nessi_logfile

Change the account (in line `#SBATCH --account=REPLACE...`) in the job script
above and store it in a file named `local_gromacs.sh`
Then, submit it with

    sbatch local_gromacs.sh

When the job has ended it should have produced a file `slurm-JOBID.out` where
`JOBID` is the id of the job that was run.

**Please, do experiment with the examples above. For example,**

- In a single job script, execute the software twice (line with `time gmx ...`)
  to observe different runtimes.
- Replace `GROMACS` with another software that is provided by NESSI. All
  available packages can be listed with
  ```
  module avail
  ```
- Use EESSI instead of NESSI. You'll likely observe different start-up times
  because NESSI and EESSI are served from different distribution networks.

**Additional information**

- Read about [future topics to be added to the documentation about NESSI](nessi-eessi-future-topics).
- Read more about EESSI at [eessi.io/docs](https://eessi.io/docs).
