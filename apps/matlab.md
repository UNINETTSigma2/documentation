# MATLAB

MATLAB is a platform for solving engineering, mathematical, and graphical problems.

To find out more, visit the MATLAB website at: http://se.mathworks.com/help/matlab

## License Information
#### Academic users
Academic users need a link to a MATLAB licence server for their university (UiB, UiT, UiO or NTNU).
Users from other universities can also use MATLAB on Fram. Send an email to support@metacenter.no 
and ask for the link to your university. Add this link to the environment variabel MLM_LICENCE_FILE:

    export MLM_LICENSE_FILE=<link-to-matlab-licens-server>
    
Add this environment variable setting into .bashrc

#### Commercial users
Commercial users need to sign a Hosting Provider agreement. Contact: sigma2@uninett.no

## Running MATLAB

| Module     | Version     |
| :------------- | :------------- |
| MATLAB |2017a|
| MATLAB |2018a|

To see available versions when logged into Fram issue command

    module spider matlab
    
To use MATLAB type

    module load MATLAB/<version>

specifying one of the available versions.

### Sample MATLAB Job Script
```
#!/bin/bash
#SBATCH --account=nnNNNNk
#SBATCH --job-name=jobname
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore system
module load MATLAB/<version>

matlab -nodisplay -nodesktop -nojvm -r "myprogram"

## Note: if you are using the Parallel Computing Toolbox, remove -nojvm

```

## MPI for Matlab
MPI for Matlab is installed on Fram (for parallelizing of many compute nodes)

User guide:

Distributed Matlab (for non MPI programmers): https://www.hpc.ntnu.no/pages/viewpage.action?pageId=15794234

Matlab MPI: https://www.hpc.ntnu.no/display/hpc/Matlab+MPI

