# LAMMPS

LAMMPS is a classical molecular dynamics code, and an acronym for Large-scale
Atomic/Molecular Massively Parallel Simulator. LAMMPS has potentials for
solid-state materials (metals, semiconductors) and soft matter (biomolecules,
polymers) and coarse-grained or mesoscopic systems.

It can be used to model atoms or, more generically, as a parallel particle
simulator at the atomic, meso, or continuum scale. LAMMPS runs on single
processors or in parallel using message-passing techniques and a
spatial-decomposition of the simulation domain. The code is designed to be easy
to modify or extend with new functionality. LAMMPS is distributed as an open
source code under the terms of the GPL.

[More information here.](https://www.lammps.org)

## Running LAMMPS

| Module     | Version     |
| :------------- | :------------- |
| LAMMPS |11Aug17-foss-2017a <br> 13Mar18-foss-2018a <br>|

To see available versions when logged into Fram issue command

    module spider lammps

To use LAMMPS type

    module load LAMMPS/<version>

specifying one of the available versions.

## Running on GPUs

```{note}
We are working on compiling the LAMMPS module on Saga and Betzy with GPU
support, however, since that could take some time we are, for the time being,
presenting an alternative solution below.
```

LAMMPS is capable of running on GPUs using the [`Kokkos`
framework](https://github.com/kokkos/kokkos). However, the default distribution
of LAMMPS on Saga and Betzy are not compiled with GPU support. We will therefor
use {ref}`singularity<running-containers>` to use LAMMPS with GPUs.

### Preparations

We will first start by creating our LAMMPS singularity image based on [Nvidia's
accelerated images](https://catalog.ngc.nvidia.com/orgs/hpc/containers/lammps).
The following command will download the image and create a singularity
container with the name `lammps.sif`.

```console
[user@login-X.SAGA ~]$ singularity pull --name lammps.sif docker://nvcr.io/hpc/lammps:29Sep2021
```

Then, to retrieve the input test file (`in.lj.txt`) execute the following:

```console
[user@login-X.SAGA ~]$ wget https://lammps.sandia.gov/inputs/in.lj.txt
```

### Slurm script

To run this we will use the following Slurm script:

```sh
#!/bin/bash                                                                        

#SBATCH --job-name=LAMMPS-Singularity
#SBATCH --account=nn<XXXX>k
#SBATCH --time=10:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks-per-node=4
#SBATCH --partition=accel
#SBATCH --gpus=2

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
# Note: We don't need any additional modules here as Singularity is always
# available

srun singularity run --nv -B ${PWD}:/host_dir lammps.sif\
  lmp -k on g 2 -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8\
  -var x 8 -var y 8 -var z 8 -in /host_dir/in.lj.txt
```

In the script above note that `--gpus=X` (or `--gpus-per-node=X`) needs to be
the same as the parameter `-k on g X` for LAMMPS. We also recommend that users
have one MPI rank per GPU so that `--ntasks=X` is either equal to `--gpus=X` or
use `--gpus-per-task=1`.

### Performance increase

We modified the above input file to run for a bit longer (increased the number
of steps to `1000`). This gave us the following speed-ups compared to
`LAMMPS/3Mar2020-foss-2019b-Python-3.7.4-kokkos` on Saga.

| Node configuration | Performance (`tau/day`) | Speed up |
|--------------------|-------------------------|----------|
| 40 CPU cores | 1439.286 | 1.0x |
| 1 GPU | 2670.089 | 1.8x |
| 2 GPUs | 4895.459 | 3.4x |
| 4 GPUs | 11610.741 | 8.0x |

## License Information

LAMMPS is available under the GNU Public License (GPLv3). For more information,
visit [the LAMMPS documentation
pages](https://docs.lammps.org/Intro_opensource.html).

It is the user's responsibility to make sure they adhere to the license
agreements.

## Citation

When publishing results obtained with the software referred to, please do check
the [developers web page in order to find the correct
citation(s)](https://docs.lammps.org/Intro_citing.html).
