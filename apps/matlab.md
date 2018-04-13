# MATLAB

MATLAB is a platform for solving engineering, mathematical, and graphical problems.

To find out more, visit the MATLAB website at: http://se.mathworks.com/help/matlab

## Running MATLAB

To load a MATLAB module, run in the terminal

    module load MATLAB/<version>

Run `module avail` to see the complete list of available versions. The table below lists the
available versions.


| Module     | Version     |
| :------------- | :------------- |
| MATLAB |2017a|

## Licensing
### Academic users:
Academic users need a link to a MATLAB licence server for their university (UiB, UiT, UiO or NTNU).
Users from other universities can also use MATLAB on Fram.
Send an email to support@metacenter.no and ask for the link to your university.
Add this link to the environment variabel MLM_LICENCE_FILE:

    export MLM_LICENSE_FILE=<link-to-matlab-licens-server>

Add this environment variable setting into .bashrc

### Commercial users:
Commercial users need to sign a Hosting Provider agreement. Contact: sigma2@uninett.no

## Sample MATLAB Job Script
```
#!/bin/bash
#SBATCH --account=my_account
#SBATCH --job-name=jobname
#SBATCH --time=0:30:0
#SBATCH --nodes=1
#SBATCH --qos=preproc
## Software modules
module restore system
module load MATLAB/2017a
module list
matlab -nodisplay -nodesktop -nojvm -r "myprogram"

```
(Note! If you are using Parallel Computing Toolbox, remove -nojvm)

To run the job: sbatch job.sh

## MPI for Matlab
MPI for Matlab is installed on Fram (for parallelizing of many compute nodes)

User guide:

Distributed Matlab (for non MPI programmers): https://www.hpc.ntnu.no/pages/viewpage.action?pageId=15794234

Matlab MPI: https://www.hpc.ntnu.no/display/hpc/Matlab+MPI

