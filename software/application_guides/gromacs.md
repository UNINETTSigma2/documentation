# GROMACS


GROMACS is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles.

More information: http://www.gromacs.org

## Running GROMACS

| Module     | Version     |
| :------------- | :------------- |
| GROMACS |2016.3-foss-2017a <br>2016.3-intel-2017a <br>2016.5-intel-2018a <br>|

To see available versions when logged into Fram issue command

    module spider gromacs

To use GROMACS type

    module load GROMACS/<version>

specifying one of the available versions in the table above.

### Sample GROMACS Job Script

```
#!/bin/bash
#SBATCH --account=nnNNNNk
#SBATCH --job-name=topol
#SBATCH --time=1-0:0:0
#SBATCH --nodes=10

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore system
module load GROMACS/<version>

case=$SLURM_JOB_NAME

## Prepare input files
cp $case.tpr $SCRATCH
cd $SCRATCH

mpirun gmx_mpi mdrun $case.tpr

## Copy results back to the submit directory
cleanup "cp $SCRATCH/* $SLURM_SUBMIT_DIR"
```

## License Information

GROMACS is available under the [GNU Lesser General Public License (LGPL)](http://www.gnu.org/licenses/lgpl-2.1.html), version 2.1.

It is the user's responsibility to make sure they adhere to the license agreements.

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).
