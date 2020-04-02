# NWChem

The North Western program system for computational Chemistry (NWChem) is ab initio computational chemistry software package, which also includes quantum chemical and molecular dynamics functionality.

More information: http://www.nwchem-sw.org


## Running NWChem

| Module     | Version     |
| :------------- | :------------- |
| NWChem |6.6.revision27746-intel-2017a-2015-10-20-patches-20170814-Python-2.7.13<br> 6.8.revision-v47-intel-2018a-2017-12-14-Python-2.7.14|

To see available versions when logged into Fram issue command

    module spider nwchem

To use NWChem type

    module load NWChem/<version>

specifying one of the available versions.

### Sample NWChem Job Script
```
#!/bin/bash
#SBATCH --account=nnNNNNk
#SBATCH --job-name=tce_benzene_2emet_1
#SBATCH --time=1-0:0:0
#SBATCH --nodes=10

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore system
module load NWChem/<version>

case=$SLURM_JOB_NAME

## Prepare input files
cp $case.nw $SCRATCH
cd $SCRATCH
mkdir $SCRATCH/tmp
export SCRATCH_DIR=$SCRATCH/tmp

mpirun nwchem $case.nw

## Copy results back to the submit directory
cleanup "cp $SCRATCH/* $SLURM_SUBMIT_DIR"
```

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).
