# How to Run Mathematica on Saga and Fram Using Singularity

## Introduction

To execute Mathematica tasks on High-Performance Computing (HPC) systems like Fram or Saga, we employ Singularity, a container solution. This guide focuses on a non-graphical, command-line interface for Mathematica. The non-graphical approach is well-suited for HPC environments because it integrates easily with job scripts in the SLURM queuing system, optimizing resource utilization and allowing for automated, batch processing of tasks.

## Pre-requisites

1. SSH access to Fram or Saga HPC systems
2. A valid Mathematica license
3. Basic shell command knowledge

## Steps

### Step 1: Pull the Container

On the HPC terminal, execute:

```bash
singularity pull docker://wolframresearch/wolframengine
```
This should download a file named `wolframengine_latest.sif`.


### Step 2: Create License File

Create a file named `mathematica-license.txt` and include the line:

```bash
!license-server.example.org
```

Replace `license-server.example.org` with your specific license details.

### Step 3:  Run a "Hello" Statement

The point of this step is to test that the installation works and the license is available. To run a Mathematica statement that prints "Hello", execute the following:

``` bash
singularity run --bind path/mathematica-license.txt:/root/.WolframEngine/Licensing/mathpass wolframengine_latest.sif wolframscript -code 'Print["Hello"]'
```

## Your first mathematica job

To create the job script we recommend using the [Slurm job script generator](https://open.pages.sigma2.no/job-script-generator/.https://open.pages.sigma2.no/job-script-generator/). The following is an example of a job script that runs a Mathematica script named `mathematica-script.wl` on Fram :

```bash
#!/bin/bash
# file-name: first-mathematica-job.sh
#SBATCH --job-name=first-mathematica-job   ## Name of the job
#SBATCH --output=slurm-%j.out   ## Name of the output-script (%j will be replaced with job number)
#SBATCH --account=nnxxxxk   ## The billed account
#SBATCH --time=00:01:00   ## Walltime of the job
#SBATCH --ntasks=1   ## Number of tasks that will be allocated
#SBATCH --nodes=1   ## Number of nodes that will be allocated

set -o errexit   ## Exit the script on any error
set -o nounset   ## Treat any unset variables as an error

CONTAINER= # Path to the container, e.g., /cluster/projects/nnxxxxk/username/containers/wolframengine_latest.sif
LICENSE= # Path to the license file, e.g., /cluster/home/username/mathematica-license.txt

singularity run --bind $LICENSE:/root/.WolframEngine/Licensing/mathpass $CONTAINER wolframscript -file mathematica-script.wl
```

```mathematica
(*mathematica-script.wl*)
Print["Hello, from Mathematica"]
```
To submit the job, execute:

```bash
sbatch first-mathematica-job.sh
```
