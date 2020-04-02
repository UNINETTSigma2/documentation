# NAMD
NAMD is designed for simulating large biomolecular systems. NAMD scales to many cores and is compatible with AMBER.

More information: http://www.ks.uiuc.edu/Research/namd

## Running NAMD

| Module     | Version     |
| :------------- | :------------- |
| NAMD |2.12-foss-2017a-mpi <br>2.12-intel-2018a-mpi <br>2017-11-06-foss-2017a-mpi <br>2018-02-15-intel-2018a-mpi <br>|

To see available versions when logged into Fram issue command

    module spider namd

To use NAMD type

    module load NAMD/<version>

specifying one of the available versions.

### Sample NAMD Job Script
```
#!/bin/bash
#SBATCH --account=nnNNNNk
#SBATCH --job-name=ubq_ws
#SBATCH --time=1-0:0:0
#SBATCH --nodes=10

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore system
module load NAMD/<version>

case=$SLURM_JOB_NAME

## Prepare input files
cp $case.* $SCRATCH
cp par_all27_prot_lipid.inp $SCRATCH
cd $SCRATCH

mpirun namd2 $case.conf

## Copy results back to the submit directory
cleanup "cp $SCRATCH/* $SLURM_SUBMIT_DIR"
```

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).
