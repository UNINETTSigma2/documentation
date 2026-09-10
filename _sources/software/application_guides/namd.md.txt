# NAMD
NAMD is designed for simulating large biomolecular systems. NAMD scales to many cores and is compatible with AMBER.

[More information here.](https://www.ks.uiuc.edu/Research/namd/)

## Running NAMD

| Module     | Version     |
| :------------- | :------------- |
| NAMD |2.12-foss-2017a-mpi <br>2.12-intel-2018a-mpi <br>2017-11-06-foss-2017a-mpi <br>2018-02-15-intel-2018a-mpi <br>|

To see available versions when logged into the machine in question, type command

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

## GPU acceleration

NAMD optionally supports acceleration on Saga and Betzy, using the attached
Nvidia GPUs of those systems. For users, very little adaptation is required,
mostly focused on the launch parameters needed (see below). To use the GPU
accelerated NAMD library, load any version of NAMD with the `fosscuda`
toolchain. Unfortunately this toolchain does not support MPI so if your problem
benefits more from wide scaling this is not applicable (please contact us
{ref}`if this applies to you <support-line>`).

```{note}
NAMD can utilize multiple GPUs, on a single node, but can also benefit from
running with multiple threads per GPU. Therefore we recommend testing with the
`--gpus=X` flag when selecting GPUs so that the number of threads are
independent from the number of GPUs requested.
```

### Example Slurm GPU script

```bash
#!/usr/bin/bash

#SBATCH --job-name=apoa1
#SBATCH --account=nn<NNNN>k
#SBATCH --time=00:30:00

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G

#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet reset  # Reset the modules to the system default
module load NAMD/2.14-fosscuda-2019b
module list

namd2 +p$SLURM_CPUS_PER_TASK +devices $CUDA_VISIBLE_DEVICES apoa1.namd
```

### Potential speed-up

We ran the [`APOA1`
benchmark](https://www.ks.uiuc.edu/Research/namd/utilities/apoa1/) on Saga
comparing a full node (40 CPUs) against a few GPU configurations and got the
following performance.

| Node configuration | Wall time (s) | Speed up |
|--------------------|---------------|----------|
| 40 CPU cores (MPI) | 235 | 1x |
| 1 CPU core + 1 GPU | 181 | 1.3x |
| 8 CPU cores + 1 GPU | 60 | 3.9x |
| 8 CPU cores + 2 GPUs | 56 | 4.2x |
| 24 CPU cores + 4 GPUs | 33 | 7.1x |

Note that depending on your setup you might not see the same performance, we
urge researchers to test with GPU to see if they can benefit and
{ref}`contact us for assistance in getting started <support-line>` if necessary.

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).
