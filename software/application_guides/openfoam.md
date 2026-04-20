# OpenFOAM

OpenFOAM is a computational fluid dynamics (CFD) software suite for physical and
engineering simulations. The suite consists of mechanical, electronics, and
embedded software simulation applications.
[To find out more, visit the OpenFOAM website.](https://www.openfoam.com/)

(openfoam_license_information)=
## License Information

OpenFOAM is available under the [GNU General Public License](https://www.gnu.org/licenses/gpl.html) (GPL). For more information, visit http://www.openfoam.com/legal/open-source.php

It is the user's responsibility to make sure they adhere to the license agreements.

1. [Running OpenFOAM](running_openfoam)
2. [OpenFOAM Container Build and Job Script Guide](openfoam-container-build-and-job-script-guide)
3. [Guide to Creating and Compiling a Custom OpenFOAM Solver](guide-to-creating-and-compiling-a-custom-openfoam-solver)




(running_openfoam)=
## Running OpenFOAM with our Module System

To see available versions when logged into a cluster, type

    module spider openfoam

To use OpenFOAM type

    module load OpenFOAM/<version>

specifying one of the available versions.

### Sample OpenFOAM Job Script

```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=your_job_name
#SBATCH --time=0-1:0:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32

module reset
module load OpenFOAM/<version>
source $FOAM_BASH

blockMesh > log.blockMesh 2>&1
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
decomposePar > log.decomposePar 2>&1
mpirun rhoSimpleFoam -parallel
reconstructPar
```

```{note}
Users has reported that the SnappyHexMesh may hang when using the Toolchains on our clusters. The solution is to either run in locally
or use an OpenFOAM container. For an example of a OpenFOAM container and workflow see below in this guide.
```

### Note on `decomposeParDict`

When running OpenFOAM on a cluster, ensure that your `decomposeParDict` file is configured correctly for parallel execution. Here are some key changes you might need to make:

- **Number of Subdomains**: Set `numberOfSubdomains` to match the total number of tasks (`nodes * ntasks-per-node`). For example, if you have 2 nodes and 32 tasks per node, set it to 64.

  ```
  numberOfSubdomains 64;
  ```

- **Method**: Choose a decomposition method that suits your case, such as `simple`, `scotch`, or `hierarchical`.

  ```
  method          scotch;
  ```

- **Coefficients**: If using `simple` or `hierarchical`, specify the `coeffs` to define how the domain is split.

  ```
  simpleCoeffs
  {
      n               (4 4 4); // Adjust based on your domain and number of subdomains
      delta           0.001;
  }
  ```

Ensure these settings align with your computational resources and case requirements.



(openfoam-container-build-and-job-script-guide)=
# OpenFOAM Container Build and Job Script Guide

For a detailed guide on building and running containers on HPC clusters, see
[container with build environment](container_with_build_environment).
This guide assumes you will build the container locally and upload it to the
cluster. Ensure [Apptainer](https://apptainer.org/) is installed on your machine. Contact our
[support team](support-line) for assistance.

## Step-by-Step Instructions

### 1. Create a Definition File

Save the following as `openfoam.def` on your local machine:

```
Bootstrap: docker
From: ubuntu:latest

%post
    apt-get update && apt-get install -y \
        curl sudo apt-transport-https ca-certificates gnupg lsb-release
    curl https://dl.openfoam.com/add-debian-repo.sh | bash
    apt-get update
    apt-get install -y openfoam2312-default | tee /root/openfoam_install.log
    if [ ! -d "/usr/lib/openfoam/openfoam2312" ]; then
        echo "OpenFOAM installation failed" >> /root/openfoam_install.log
        exit 1
    fi
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%environment
    source /usr/lib/openfoam/openfoam2312/etc/bashrc

%runscript
    echo "Starting OpenFOAM shell session..."
    exec /bin/bash --rcfile /usr/lib/openfoam/openfoam2312/etc/bashrc
```

### 2. Build the Container

Run:

```bash
sudo apptainer build openfoam.sif openfoam.def
```

### 3. Upload the Container to the Cluster

Use `rsync`:

```bash
rsync --info=progress2 -a openfoam.sif username@cluster:receiving-directory
```

For more information on `rsync`, see the [File Transfer](file-transfer) guide.

### 4. Create a Job Script

Save as `jobscript.sh`:

```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=your_job_name
#SBATCH --time=0-1:0:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32

module reset
APPTAINER_IMAGE=/path/to/your/openfoam.sif

apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE blockMesh > log.blockMesh 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE decomposePar > log.decomposePar 2>&1
srun --mpi=pmix_v3 apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE rhoSimpleFoam -parallel
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE reconstructPar
```

### 5. Submit the Job Script

Run:

```bash
sbatch jobscript.sh
```
(guide-to-creating-and-compiling-a-custom-openfoam-solver)=
# Guide to Creating and Compiling a Custom OpenFOAM Solver

## Step 1: Set Up Your Environment

1. **Create a Working Directory:**

   ```bash
   mkdir -p path/to/customSolvers
   cd path/to/customSolvers
   ```

2. **Copy an Existing Solver:**
    With the module system

   ```bash
   module load OpenFOAM/<version>
   source $FOAM_BASH
   cp -r $FOAM_SOLVERS/compressible/rhoSimpleFoam ./mySolver
   ```

   or with a container:

   ```bash
   export APPTAINER_IMAGE=/path/to/your/openfoam.sif
   apptainer shell $APPTAINER_IMAGE
   cp -r $FOAM_SOLVERS/compressible/rhoSimpleFoam ./mySolver
   exit
   ```

    Note the specific solver used here may not be available in all versions of OpenFOAM. Use appropriate solvers based on your version.

## Step 2: Modify the Solver

1. **Rename the Solver File:**

   ```bash
   cd mySolver
   mv rhoSimpleFoam.C mySolver.C
   ```

2. **Edit the Solver Code:**

   Replace `rhoSimpleFoam` with `mySolver` in `mySolver.C`.

3. **Update the Make Files:**

   Edit `Make/files`:

   ```
   mySolver.C
   EXE = $(FOAM_USER_APPBIN)/mySolver
   ```

## Step 3: Compile the Solver

1. **Set Environment Variables:**

   Specify your paths:

   ```bash
   export FOAM_USER_APPBIN=/path/to/your/bin
   mkdir -p $FOAM_USER_APPBIN
   ```

2. **Compile the Solver:**

   **If using a module system:**

   ```bash
   module load OpenFOAM/<version>
   source $FOAM_BASH
   wmake
   ```

   **If using a container:**

   - Enter the container shell:

     ```bash
     apptainer shell $APPTAINER_IMAGE
     ```

   - Set the environment variable inside the container:

     ```bash
     export FOAM_USER_APPBIN=/path/to/your/bin
     ```

   - Run `wmake` inside the container:

     ```bash
     wmake
     ```

   - Exit the shell:

     ```bash
     exit
     ```

### Sample Job Script for Custom Solver

**Using Module System:**

```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=custom_solver_job
#SBATCH --time=0-1:0:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32

module reset
module load OpenFOAM/<version>

FOAM_USER_APPBIN=/path/to/your/bin

blockMesh > log.blockMesh 2>&1
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
decomposePar > log.decomposePar 2>&1
mpirun $FOAM_USER_APPBIN/mySolver.C -parallel
reconstructPar
```

**Using Container:**

```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=your_job_name
#SBATCH --time=0-1:0:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32

module reset
APPTAINER_IMAGE=/path/to/your/openfoam.sif
FOAM_USER_APPBIN=/path/to/your/bin

apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE blockMesh > log.blockMesh 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE decomposePar > log.decomposePar 2>&1
srun --mpi=pmix_v3 apptainer exec --bind $(pwd):/mnt,$FOAM_USER_APPBIN:/mnt/user/bin $APPTAINER_IMAGE /mnt/user/bin/mySolver.C -parallel
apptainer exec --bind $(pwd):/mnt $APPTAINER_IMAGE reconstructPar
```
