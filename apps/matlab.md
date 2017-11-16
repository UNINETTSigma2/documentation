# MATLAB

MATLAB is a platform for solving engineering, mathematical, and graphical problems.

To find out more, visit the MATLAB website at: http://se.mathworks.com/help/matlab/

## Running MATLAB

To load the default MATLAB module, run in the terminal

    module load MATLAB/2017a

Run `module avail` to see the complete list of available versions. The table below lists the
available versions. If there are more than one, the default is marked with `*`.

To start Matlab on a login node, do as: matlab -nodisplay -nodesktop -nojvm
(Remember the license link (see License))

| Module     | Version     |
| :------------- | :------------- |
| MATLAB |2017a|

## License
You need a link to a Matlab licence server for your university (UiB,UiT,UiO or NTNU).
Send an email to support@metacenter.no and ask for the link to your university.

Add this link to the enviroment variabel MLM_LICENCE_FILE as:
export MLM_LICENSE_FILE=link-to-matlab-licens-server

Add this enviroment variable setting into your job script or .bashrc

## Job script
Job script for standard Matlab program for 1 compute node (eg myprogram.m).
Copy the text below into a file (named eg job.sh), and change --account, --job-name and license link

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
export MLM_LICENSE_FILE=link-to-license-server
matlab -nodisplay -nodesktop -nojvm -r "myprogram"

```
To run the job: sbatch job.sh