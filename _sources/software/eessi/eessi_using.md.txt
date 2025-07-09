(eessi-using)=

# Using software provided by EESSI

Once the environment is configured to
[access software provided by EESSI](eessi-access-on-nris), all a
user has to do is loading module, for example:

    module load GROMACS/2024.1-foss-2023b


**Example job script using EESSI software**

We demonstrate the use of EESSI in a job script using GROMACS as an example. Below is a simple job script. 

    #!/bin/bash
    #SBATCH --account=REPLACE_WITH_YOUR_NN****K
    #SBATCH --job-name=EESSI_GROMACS
    #SBATCH --partition=normal
    #SBATCH --mem=16G
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=2
    #SBATCH --time=5 # short runtime

    module load EESSI/2023.06
    module load GROMACS/2024.1-foss-2023b

    # prepare test data if not yet present
    if [ ! -f GROMACS_TestCaseA.tar.gz ]; then
      curl -OL https://repository.prace-ri.eu/ueabs/GROMACS/1.2/GROMACS_TestCaseA.tar.gz
    fi
    if [ ! -f ion_channel.tpr ]; then
      tar xfz GROMACS_TestCaseA.tar.gz
    fi

    rm -f ener.edr eessi_logfile.log

    # run GROMACS, we use 'time' to measure how long it runs
    # note: downscaled to just 1k steps (full run is 10k steps)
    time gmx mdrun -ntmpi 2 -ntomp 2 -s ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 1000

Change the account (in line `#SBATCH --account=REPLACE...`) in the job script
above and store it in a file named `eessi_gromacs.sh`
Then, submit it with

    sbatch eessi_gromacs.sh

When the job has ended it should have produced a file `slurm-JOBID.out` where
`JOBID` is the id of the job that was run.


**Example job script using locally installed software**

We demonstrate the minimal change needed to the run the same software
(`GROMACS`) using the installations provided locally on the HPC systems the
following script just comments out the line loading EESSI module (`# module load EESSI/2023.06`) and load a locally available GROMACS module

    #!/bin/bash
    #SBATCH --account=REPLACE_WITH_YOUR_NN****K
    #SBATCH --job-name=LOCAL_GROMACS
    #SBATCH --partition=normal
    #SBATCH --mem=16G
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=2
    #SBATCH --time=5 # short runtime

    # module load EESSI/2023.06
    module load GROMACS/2023.1-foss-2022a

    # prepare test data if not yet present
    if [ ! -f GROMACS_TestCaseA.tar.gz ]; then
      curl -OL https://repository.prace-ri.eu/ueabs/GROMACS/1.2/GROMACS_TestCaseA.tar.gz
    fi
    if [ ! -f ion_channel.tpr ]; then
      tar xfz GROMACS_TestCaseA.tar.gz
    fi

    rm -f ener.edr eessi_logfile.log

    # run GROMACS, we use 'time' to measure how long it runs
    # note: downscaled to just 1k steps (full run is 10k steps)
    time gmx mdrun -ntmpi 2 -ntomp 2 -s ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 1000

Change the account (in line `#SBATCH --account=REPLACE...`) in the job script
above and store it in a file named `local_gromacs.sh`
Then, submit it with

    sbatch local_gromacs.sh

When the job has ended it should have produced a file `slurm-JOBID.out` where
`JOBID` is the id of the job that was run. For the examples above, the `slurm-JOBID.out`
file contains the following information (among other things):
- which program was actually run, e.g., GROMACS conveniently reports somthing like

      Executable:   /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/software/GROMACS/2024.1-foss-2023b/bin/gmx

  Note, when using EESSI, the exact path will be specific to the
  compute node the job was running on, because for each machine the version
  optimised for that CPU architecture is automatically selected.

- how well GROMACS was run, e.g., it prints `Performance` information (higher
  values for `ns/day` are better, for `hours/ns` lower values are better),

- finally, the job statistics at the end show how long the overall job was run
  (see `Elapsed wallclock time`) and how well the job used the reserved resources
  (see `Used CPU time` and `Unused CPU time`)

**Please, do experiment with the examples above. For example,**

- In a single job script, execute the software twice (line with `time gmx ...`)
  to observe different runtimes.
- Replace `GROMACS` with another software that is provided by EESSI. All
  available packages can be listed with
  ```
  module avail
  ```

**Additional information**

- Read about [more information about EESSI](eessi-topics).
- Read more about EESSI at [eessi.io/docs](https://eessi.io/docs).
