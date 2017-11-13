# MATLAB

MATLAB is a platform for solving engineering, mathematical, and graphical problems.

To find out more, visit the MATLAB website at: http://se.mathworks.com/help/matlab/

## Running MATLAB

To load the default MATLAB module, run in the terminal

    module load MATLAB

Run `module avail` to see the complete list of available versions. The table below lists the
available versions. If there are more than one, the default is marked with `*`.

| Module     | Version     |
| :------------- | :------------- |
| MATLAB |2017a|

## License
You need a link to a Matlab licence server for your university (UiB,UiT,UiO and NTNU).
Send an email to support@metacenter.no and ask for this link.
Add the link in you job script or in .bashrc as:

export MLM_LICENSE_FILE=link-to-matlab-licens-server

## Job script
Standard matlab program (eg. myprogram.m)

```
#!/bin/bash
#SBATCH --account=my_account
#SBATCH --job-name=jobname
#SBATCH --time=0:30:0
#SBATCH --nodes=1
## Software modules
module restore system
module load MATLAB/2017a
module list
export MLM_LICENSE_FILE=link-to-license-server
matlab -nodisplay -nodesktop -nojvm -r "myprogram"

```