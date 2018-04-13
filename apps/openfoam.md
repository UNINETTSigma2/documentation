# OpenFOAM
OpenFOAM is a computational fluid dynamics (CFD) software suite for physical and
engineering simulations. The suite consists of mechanical, electronics, and
embedded software simulation applications.

To find out more, visit the OpenFOAM website at: http://www.openfoam.com/

# Running OpenFOAM

To load a OpenFOAM module, run in the terminal

    module load OpenFOAM/<version>

Run `module avail` to see the complete list of available versions. The table below lists the
available versions.

| Module     | Version     |
| :------------- | :------------- |
| OpenFOAM |4.1-intel-2017a <br>5.0-intel-2017a <br>|
| OpenFOAM-Extend |4.0-intel-2017a |

## Sample OpenFOAM Job Script
```
#!/bin/bash

#SBATCH --job-name=damBreak
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --account=nnNNNNk

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore system
module load OpenFOAM/1712-foss-2018a
source $FOAM_BASH

case=$SLURM_JOB_NAME

cp -r $case/* $SCRATCH
cd $SCRATCH

mpirun interFoam -parallel

reconstructPar

mkdir -p $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
cleanup "cp -r $SCRATCH/constant $SCRATCH/system $SCRATCH/[0-9]* $SLURM_SUBMIT_DIR/$SLURM_JOB_ID"

```

# License Information

OpenFOAM is available under the [GNU General Public License](https://www.gnu.org/licenses/gpl.html) (GPL). For more information, visit http://www.openfoam.com/legal/open-source.php

It is the user's responsibility to make sure they adhere to the license agreements.
