# MATLAB

MATLAB is a platform for solving engineering, mathematical, and graphical problems.

To find out more, visit the MATLAB website at: http://se.mathworks.com/help/matlab

## License Information

### Academic users

Academic users need a link to a MATLAB license server for their university
(UiB, UiT, UiO or NTNU).  Users from other universities can also use MATLAB on
Fram. Send an email to support@metacenter.no and ask for the link for your
university and the license name to use when submitting jobs. Add this link to
the environment variable MLM_LICENSE_FILE:

    export MLM_LICENSE_FILE=<link-to-matlab-license-server>

Add this environment variable setting into your `~/.bashrc`.  When submitting
a job with, e.g., sbatch, use `sbatch --licenses=<license-name>`.


#### Third-Party Access for Collaborative Research in Academia
See this link: https://se.mathworks.com/support/collaborative-research-academia.html

#### Commercial users
Commercial users need to sign a Hosting Provider agreement. Contact: sigma2@uninett.no

## Running MATLAB

| Module     | Version     |
| :------------- | :------------- |
| MATLAB |2017a|
| MATLAB |2018a|
| MATLAB |2018b|
| MATLAB |2019a|
| MATLAB |2020b|

To see available versions when logged into Fram issue command

    module spider matlab

To use MATLAB type

    module load MATLAB/<version>
    (eg. module load MATLAB/2020b)

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
MPI for Matlab is installed on Fram (for parallelizing on many compute nodes)

User guide:

Distributed Matlab (for non MPI programmers): https://www.hpc.ntnu.no/pages/viewpage.action?pageId=15794234

Matlab MPI: https://www.hpc.ntnu.no/display/hpc/Matlab+MPI
