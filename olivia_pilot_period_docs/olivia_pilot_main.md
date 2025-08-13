---
orphan: true
---


(olivia_pilot_main)=

# Olivia Pilot Info

<!-- **Olivia User Guide ** (Skeleton draft) -->

```{danger}
__!! This page is currently under development and work in progress. Configs/information may not be final !!__
```

```{contents} __Page Overview__
:depth: 3
```

## Duration of pilot period
The pilot period is expected to start on Monday 7 July and will last until 21 September 2025.

## How to connect to Olivia
Logging into Olivia involves the use of {ref}`Secure Shell (SSH) <ssh>` protocol,
either in a terminal shell or through a graphical tool using this protocol
under the hood.

In the future, we will introduce the same 2 factor authentication (2FA) system
as used on our other HPC machines. But for now, you cannot connect directly
from your local machine. Instead you have to __first connect to Betzy, Fram or
Saga and then further to Olivia__.

Replace `<username>` with your registered username:

```console
$ ssh <username>@betzy.sigma2.no
$ ssh <username>@olivia.sigma2.no
```

## System architecture and technical spec

Olivia is based on HPE Cray EX platforms for compute and accelerator nodes, and HPE ClusterStor for global storage, all connected via HPE Slingshot high-speed interconnect.

### CPU Compute Nodes

The CPU-based compute resources consist of **252 nodes**. Each node features two **AMD EPYC Turin 128-core CPUs (Zen5c architecture)**, with **768 GiB of DDR5-6000 memory (24x32GB DIMMs)**. Each node also includes **3.84 TB of solid-state local storage (M.2 SSD)**. All Class A nodes are **direct liquid cooled**.

### Accelerator Nodes

These nodes are optimized for accelerated HPC and AI/ML workloads, totaling **76 nodes with 304 NVIDIA GH200 superchips**. Each GH200 device has **96 GB HBM3 memory (GPU)** and **120 GB LPDDR5x memory (CPU)**, connected via a **900 GB/s bidirectional NVlink C2C connection**. These nodes are also **direct liquid cooled**. Their dedicated storage is provided by a **HPE Cray C500 flash-based parallel filesystem with a total capacity of 510 TB**.


### Service Nodes

Olivia has **8 service nodes**, with 4 nodes having both CPU and GPU (**NVIDIA L40 GPUs** for remote visualization and compilation). Each node has **2 AMD EPYC 9534 64-core processors** and **1536 GiB DDR5-4800 memory**. Storage per node includes **2x 960 GB NVMe SSDs for the system drive** (configured as RAID mirror for redundancy) and **4x 6.4 TB NVMe SSDs for data (19.2 TB total)**. These nodes are **air-cooled**.

### I/O Nodes

Olivia has **5 dedicated nodes for I/O operations**, based on HPE ProLiant DL385 Gen 11. Each node has **2 AMD EPYC Genoa 9534 64-core CPUs** and **512 GiB memory**. They are designed to connect to the NIRD storage system and the Global Storage simultaneously. These nodes are **air-cooled**.

### High-Performance Global Storage

This system, based on ClusterStor E1000, offers **4.266 PB of total usable capacity**, including **1.107 PB of solid-state storage (NVMe)** and **3.159 PB of HDD-based storage**. It is POSIX compatible and can be divided into several file systems (e.g., scratch, home, projects, software). It supports RDMA and is designed for fault tolerance against single component failures. The system is **air-cooled**.

### High-Speed Interconnect

The interconnect uses **HPE Slingshot**, a Dragonfly topology that supports **200 Gbit/s simplex bandwidth** between nodes. It natively supports IP over Ethernet, and RDMA or RoCE access to the storage solution.





## Project accounting (can be part of general accounting)
**Invoicing**: During the piloting phase there will no invoicing for usage of the system.

Project compute quota usage is accounted in 'billing units' BU. The calculation of elapsed BU for a job is done in the same manner as elsewhere on our systems.

The calculation of elapsed GPU-hours is currently only dependent on the number of GPUs allocated for a job.


## Software

Olivia introduces several new aspects that set it apart from our current HPC systems.

First and foremost, the large number of GPU nodes and the software that will run on them represent uncharted territory for us, as we don't yet have extensive experience in this area. Additionally, Olivia is an HPE Cray machine that leverages the [Cray Programming Environment (CPE)](https://cpe.ext.hpe.com/docs/latest/getting_started/CPE-General-User-Guide-HPCM.html), which comes with its own suite of tools for development, compilation, and debugging. As a result, we cannot simply replicate the software stack from our other systems. Instead, we need to adapt, modify, and expand the ways in which software is installed and utilized.

The pilot phase will serve as a testing ground for the approaches outlined below, helping us identify the best solutions for providing software on Olivia. Please note that __things will evolve during the pilot phase__, and the final configuration may differ significantly from what is described here. Your feedback is invaluable—if something doesn't work or could be improved, let us know so we can make adjustments.

```{danger}
__Python and PyTorch user__

Please read the sections 2 and 3 below.
Do not use Anaconda or pip directly to install large environments with many packages.
```

### 1. Module System

The module system provides a convenient way to access software packages and libraries that are installed and maintained specifically for the Olivia HPC cluster. All software is compiled and optimized for Olivia's nodes to ensure the best performance.

#### CPU and GPU Architectures

Olivia features three distinct CPU architectures:

- **Login Nodes**: Use x86-64 processors with the Zen4 microarchitecture.
- **CPU Compute Nodes**: Use x86-64 processors with the Zen5 microarchitecture.
- **GPU Nodes**: Use ARM-based Neoverse V2 processors.

To accommodate these architectures, we have prepared three separate software stacks. You can initialize the module system and access these stacks by running the following commands:

```bash
$ module purge
$ source /opt/cray/pe/lmod/lmod/init/profile
$ export MODULEPATH=/cluster/software/modules/Core/
```

After initialization, you can list the available stacks using:

```bash
$ module available
```

You should see output similar to this:

```
--------------------- /cluster/software/modules/Core -----------
   BuildEnv/NeoverseV2    BuildEnv/Zen5 (D)    NRIS/GPU   (S)
   BuildEnv/Zen4          NRIS/CPU      (S)    NRIS/Login (S,D)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
   D:  Default Module
```

If the list of available modules is too long, you can use the following commands for a more focused view:

- `module --default avail` or `ml -d av`: Lists only the default modules.
- `module overview` or `ml ov`: Displays the number of modules for each name.

To search for specific modules or extensions, use:

- `module spider`: Finds all possible modules and extensions.
- `module keyword key1 key2 ...`: Searches for modules matching specific keywords.

---

#### NRIS Software Stacks

The `NRIS` modules provide preinstalled software tailored for different node types:

- **`NRIS/CPU`**: Contains software packages and libraries optimized for the **CPU compute nodes**.
- **`NRIS/GPU`**: Currently includes libraries, compilers, and tools for building software for the Grace-Hopper 200 GPUs. In the future, this stack will also include AI frameworks such as PyTorch and TensorFlow (see _AI Frameworks_ below).
- **`NRIS/Login`**: Includes tools for pre- and post-processing data.
  **Note**: Do not use this stack for running workflows on the compute nodes.

To load a stack and view its available software, use:

```bash
$ module load NRIS/CPU
$ module avail
```

---

##### Searching for Modules Across Stacks

You can search for a specific module across all stacks using `module spider`. For example:

```bash
$ module spider SCOTCH
```

This will display information about the `SCOTCH` module, including available versions:

```
-----------------------------------------------------------
  SCOTCH:
-----------------------------------------------------------
    Description:
      Software package and libraries for sequential and
      parallel graph partitioning, static mapping, and
      sparse matrix block ordering, and sequential mesh and
      hypergraph partitioning.

     Versions:
        SCOTCH/7.0.3-gompi-2023a
        SCOTCH/7.0.4-gompi-2023b
        SCOTCH/7.0.6-gompi-2024a
```

To get detailed information about a specific version, including how to load it, use the module's full name:

```bash
$ module spider SCOTCH/7.0.6-gompi-2024a
```

Similarly, for other modules like `NVHPC`, you can use:

```bash
$ module spider NVHPC/25.3-CUDA-12.8.0
```

This will provide details such as dependencies and additional help:

```
-----------------------------------------------------------
  NVHPC: NVHPC/25.3-CUDA-12.8.0
-----------------------------------------------------------
    Description:
      C, C++ and Fortran compilers included with the NVIDIA
      HPC SDK (previously: PGI)

    You will need to load all module(s) on any one of the
    lines below before the "NVHPC/25.3-CUDA-12.8.0" module is
    available to load.

      BuildEnv/NeoverseV2
      NRIS/GPU

    Help:
      Description
      ===========
      C, C++ and Fortran compilers included with the NVIDIA
      HPC SDK (previously: PGI)

      More information
      ================
       - Homepage: https://developer.nvidia.com/hpc-sdk/
```

---

##### Important Notes

- **Avoid Using `BuildEnv` Modules**: These modules are intended for internal use and will be hidden in the future. Please use the `NRIS` modules instead.
- __Software Requests__: We will install additional software packages in the future. If you cannot find the software you need, please contact us. Refer to our [Getting Help page](support-line) for details, and be sure to mention that your request is regarding Olivia.
- For additional help, consult the module system's built-in commands (`module help`, `module spider`, etc.) or contact the system administrators.

---

#### Using the EESSI Stack

The EESSI (European Environment for Scientific Software Infrastructure) software stack - optimized for each supported CPU architecture - already available on Betzy, Fram, and Saga is now also available on Olivia.

To load the EESSI stack, simply use:

```bash
$ source /cluster/installations/eessi/default/eessi_environment_variables_Olivia
$ source /cvmfs/software.eessi.io/versions/2023.06/init/bash
```

The second script automatically detects the CPU microarchitecture of the current machine and selects the most appropriate pre-built software provided by EESSI. This ensures optimal performance and works seamlessly across systems with varying CPU architectures, such as those available on Olivia.

Once the environment is configured to access software provided by EESSI, you can interact with it using the `module` command, just like with the NRIS stacks. For example:

```bash
$ module avail
$ module load TensorFlow/2.13.0-foss-2023a
```

The first command will list all software modules available within the EESSI stack (on the current CPU/GPU partition).
Then we load the `TensorFlow` module.

While EESSI provides a wide range of preinstalled software, you can **build** on top of EESSI either using EasyBuild through loading the EESSI-extend module:
`module load EESSI-extend` or manually (without EasyBuild) through loading either of the buildenv available modules: `module load buildenv/default-foss-2023a`
For more information see the official [EESSI documentation](https://www.EESSI.do/docs/using_EESSI/building_on_EESSI/).

##### GPU-enabled Software on EESSI

The official EESSI stack contains already some modules of popular software like GROMACS, but many are also still missing.

To get you started more quickly, we have added some local GPU-enabled software on Olivia which are not yet officially supported by [EESSI](https://www.EESSI.do/docs) or [EasyBuild](https://tutorial.easybuild.io/).

To access it, run the following command after initializing the EESSI environment on a __GPU node__:
```bash
$ export MODULEPATH=/cluster/installations/eessi/default/eessi_local/aarch64/modules/all:$MODULEPATH
```

Note: If your job requires the use of `mpirun`, please ensure you use the `srun --mpi=pmix` option for proper integration.
Sample job script using `OSU-Micro-Benchmarks/7.2-gompi-2023b` from the EESSI stack:
```bash
#!/bin/bash -e
#SBATCH --job-name=Sample
#SBATCH --account=nnXXXX
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

source /cluster/installations/eessi/default/eessi_environment_variables_Olivia
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
module load OSU-Micro-Benchmarks/7.2-gompi-2023b

srun --mpi=pmix osu_bw
```

---

### 2. Python, R, and (Ana-)Conda

Python and R are widely used in scientific computing, but they were originally designed for personal computers rather than high-performance computing (HPC) environments. These languages often involve installations with a large number of small files—Python environments, for example, can easily consist of tens or even hundreds of thousands of files. This can strain the file system, leading to poor performance and a sluggish user experience.

To address this, Olivia uses the [_HPC-container-wrapper_](https://github.com/CSCfi/hpc-container-wrapper/), a tool designed to encapsulate installations within containers optimized for HPC systems. This tool wraps Python or Conda environments inside containers, significantly reducing the number of files visible to the file system. It also generates executables, allowing you to run commands like `python` seamlessly, without needing to interact directly with the container. This approach minimizes file system load while maintaining ease of use.

---

#### Key Features of HPC-container-wrapper

The HPC-container-wrapper supports wrapping:

- **Conda installations**: Based on a [Conda environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).
- **Pip installations**: Based on a [pip requirements.txt file](https://pip.pypa.io/en/latest/reference/requirements-file-format/).
- **Existing installations on the filesystem**: To reduce I/O load and improve startup times.
- **Existing Singularity/Apptainer containers**: To hide the need for using the container runtime from the user.

---

#### Creating a New Python or Conda Environment

To create a new Python or Conda environment on Olivia, follow these steps:

1. **Load the necessary modules**:

   ```bash
   $ module purge
   $ source /opt/cray/pe/lmod/lmod/init/profile
   $ export MODULEPATH=/cluster/software/modules/Core/
   $ module load NRIS/CPU
   $ module load hpc-container-wrapper
   ```

2. **Prepare your environment file**:

   - For **pip**, create a `requirements.txt` file listing the packages you need. For example:

     ```text
     numpy==1.23.5
     scipy==1.10.1
     matplotlib
     ```

   - For **Conda**, create an `env.yml` file. For example:

     ```yaml
     channels:
       - conda-forge
     dependencies:
       - python=3.10
       - numpy
       - pandas
       - matplotlib
     ```

   Alternatively, you can export an environment from an existing setup (e.g., on your laptop or another cluster):

   - Export a Conda environment:

     ```bash
     $ conda env export -n <env_name> > env.yml
     ```

     *Note*: On Windows or macOS, add the `--from-history` flag to avoid including platform-specific dependencies.

   - Export a pip environment:

     ```bash
     $ pip freeze > requirements.txt
     ```

3. **Build the environment**:

   - For **pip**, use:

     ```bash
     $ pip-containerize new --prefix <install_dir> --slim requirements.txt
     ```

     The `--slim` argument uses a pre-built minimal Python container with a newer Python version as a base. Without `--slim`, the host system is fully available, but this may include unnecessary system installations.

   - For **Conda**, use:

     ```bash
     $ conda-containerize new --prefix <install_dir> env.yml
     ```

     If you also need to install additional pip packages, you can combine both files:

     ```bash
     $ conda-containerize new -r requirements.txt --prefix <install_dir> env.yml
     ```

4. **Add the environment to your PATH**:

   After the installation is complete, add the `bin` directory of your environment to your `PATH`:

   ```bash
   $ export PATH="<install_dir>/bin:$PATH"
   ```

   You can now call `python` or any other executables installed in the environment as if the environment were activated.

---

#### Modifying an Existing Environment

Since the environment is wrapped inside a container, direct modifications are not possible. However, you can update the environment using the `update` keyword along with a post-installation script. For example:

```bash
$ conda-containerize update <install_dir> --post-install <script.sh>
```

The `<script.sh>` file might contain commands like:

```bash
$ conda install -y seaborn
$ conda remove -y pyyaml
$ pip install requests
```

---

#### Tips and Troubleshooting

- **Exporting environments**: If you encounter issues with version conflicts, you can remove version specifications from your environment file using a simple `sed` command. For example:

  ```bash
  $ sed '/==/s/==.*//' requirements.txt > requirements_versionless.txt  # Works for pip files
  $ sed '/=/s/=.*//' env.yml > env_versionless.yml  # Works for conda files
  ```

- **Using Mamba**: For faster Conda installations, you can enable [Mamba](https://github.com/mamba-org/mamba) by adding the `--mamba` flag:

  ```bash
  $ conda-containerize new --mamba --prefix <install_dir> env.yml
  ```

- **Limitations**: Be aware of the [limitations of HPC-container-wrapper](https://github.com/CSCfi/hpc-container-wrapper#limitations), such as its experimental status and potential issues with advanced features.

---

#### Example Workflow: End-to-End Conda Installation

1. Create an `env.yml` file:

   ```yaml
   channels:
     - conda-forge
   dependencies:
     - python=3.9
     - numpy
     - scipy
     - matplotlib
   ```

2. Build the environment:

   ```bash
   $ conda-containerize new --prefix MyEnv env.yml
   ```

3. Add the environment to your `PATH`:

   ```bash
   $ export PATH="$PWD/MyEnv/bin:$PATH"
   ```

4. Use the environment:

   ```bash
   $ python --version
   Python 3.9.0
   ```

---

For more examples and advanced usage, see the [HPC-container-wrapper GitHub repository](https://github.com/CSCfi/hpc-container-wrapper) and check the [documentation page of CSC about HPC-container-wrapper](https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#wrapping-a-plain-pip-installation)


---

### 3. AI Frameworks

For AI workflows based on popular frameworks like PyTorch, JAX, or TensorFlow, we aim to deliver optimal performance by utilizing containers provided by [NVIDIA](https://catalog.ngc.nvidia.com/containers). These containers are specifically optimized for GPU workloads, ensuring excellent performance while remaining relatively straightforward to use.

```{danger}
We have done testing and have recommendations for using PyTorch using Python wheels.
If you want to use PyTorch wheels during the test phase, you can refer to this {ref}`this documentation <pytorch-wheels>`.

But please be aware that direct (not containerized) installations of pip and conda environments, put a lot of stress on the Lustre file system.
Therefore, these will not be allowed after the pilot phase. Instead you have to either use the containers and modules we provide or wrap your installation yourself imanually or using the hpc-container-wrapper as explained above.
```

---

#### Downloading Containers

You can use pre-built containers or download and convert your own. Here are the options:

1. **Pre-Built Containers**:  
   Some NVIDIA containers are already available in the Apptainer image format (`.sif`) under:  
   `/cluster/work/support/container`.

2. **Downloading Your Own Containers**:  
   You can download containers from sources like the [NVIDIA NCC catalogue](https://catalog.ngc.nvidia.com/containers) or [Docker Hub](https://hub.docker.com/). Use the following command to download and convert a container into the Apptainer format:

   ```bash
   $ apptainer build <image_name>.sif docker://<docker_image_url>
   ```

   For example, to download and convert the latest NVIDIA PyTorch container:

   ```bash
   $ apptainer build pytorch_nvidia_25.06.sif docker://nvcr.io/nvidia/pytorch:25.06-py3
   ```

   For more options, check the Apptainer build help:

   ```bash
   $ apptainer build --help
   ```

---

#### Important Notes

- **Building Your Own Containers:**
  Currently it is only possible to build containers from definition files that don't require root privileges inside the container. We are working on enabling some version of fakeroot to enable most builds in the future.
- **Running on GPU Nodes**:  
  If you plan to run these containers on GPU nodes, ensure that you download the container from a GPU node. This can be done either via a job script or an interactive job. This ensures that the container is built for the correct architecture (e.g., ARM64).  
  For details on accessing internet resources from compute nodes, see [this section](olivia_internet_proxies).

- **Managing Cache**:  
  By default, Apptainer stores its cache in your home directory, which may lead to `disk quota exceeded` errors. To avoid this, set the cache directory to your project work area:

  ```bash
  $ export APPTAINER_CACHEDIR=/cluster/work/projects/<project_number>/singularity
  ```

---

#### Running Containers

Apptainer provides several ways to run containers, depending on your needs:

1. **Run the Default Command**:  
   Use `apptainer run` to execute the default command specified in the container setup. For example:

   ```bash
   $ apptainer run <image_path>.sif <extra_parameters>
   ```

   *Example*: Running a containerized application with its default configuration.

2. **Execute Arbitrary Commands**:  
   Use `apptainer exec` to run specific commands available inside the container. For example:

   ```bash
   $ apptainer exec <image_path>.sif <command> <extra_parameters>
   ```

   *Example*: Running a Python script inside the container.

3. **Interactive Shell**:  
   Use `apptainer shell` to start an interactive shell session inside the container. For example:

   ```bash
   $ apptainer shell <image_path>.sif <extra_parameters>
   ```

   *Example*: Exploring the container environment or debugging.

---

#### Enabling GPU Support

To enable GPU support inside the container, add the `--nv` flag to your Apptainer command. This ensures that the container has access to the GPU resources. For example:

```bash
$ apptainer exec --nv /cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

This command checks if GPUs are available and prints the number of GPUs detected inside the container.

---

#### Next Steps

Currently, you need to run these containers directly using Apptainer. However, we are working on simplifying the experience by integrating these tools into the module system. This will allow you to:

- Load a module that provides executables like `python` and `torch`, eliminating the need to interact with the containers directly.
- Seamlessly use AI frameworks without worrying about container management.

Additionally, we will soon provide more information on how to best utilize the interconnect and set up job scripts for running AI workflows efficiently.

---


## Running Jobs

### Job types

For the pilot period of Olivia, the following job types are defined.

| Name   | Description       | Job limits | Max walltime | Priority |
|:------:|-------------------|:----------:|:------------:|:--------:|
| normal | default job type  |            | 7 days       | normal   |
| accel  | jobs needing GPUs |            | 7 days       | normal   |
| devel  | development jobs  |            | 2 hours      | normal   |

Note that job limits will change after pilot period is over.

### Special notes

At this point running OpenMPI jobs that use a single compute node requires a special sbatch argument:

   ```
   #SBATCH --network=single_node_vni
   ```

OpenMPI applications can be started with `mpirun` (native OpenMPI way), or `srun`. In the latter case on Olivia this has to be done in the following way

   ```
   srun --mpi=pmix <...>
   ```

Applications compiled with Intel MPI and Cray MPI should not use this extra argument to `srun`.

#### Normal jobs (CPUs)

- __Allocation units__: CPUs and memory
- __Job Limits__:
    - no limits
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__:
	- 252 nodes each with 2*128 CPUs and 768 GiB RAM
- __Parameter for sbatch/sallow__:
    - None, _normal_ is the default
- Other notes:
    - This is the default job type.
    - Normal jobs have $SCRATCH on local NVMe disk on the nodes


(job_scripts_olivia_normal)=
##### Normal job script
Job scripts for normal jobs on Olivia are very similar to the ones on
Saga.  For a job simple job running one process with 32 threads, the
following example is enough:

    #SBATCH --account=nnXXXXk     # Your project
    #SBATCH --job-name=MyJob      # Help you identify your jobs
    #SBATCH --time=1-0:0:0        # Total maximum runtime. In this case 1 day
    #SBATCH --cpus-per-task=32    # Number of CPU cores
    #SBATCH --mem-per-cpu=3G      # Amount of CPU memory per CPU core
    #SBATCH --network=single_node_vni # Currently required for OpenMPI to work with single compute node jobs.

#### Accel jobs (GPU)

*accel* jobs give access to use the H200 GPUs.

- __Allocation units__: CPUs, memory and GPUs
- __Job Limits__:
    - no limits
- __Maximum walltime__: 7 days
- __Priority__: normal
- __Available resources__: 76 nodes each with 4*72 ARM CPU cores, 4*120 GiB CPU RAM and 4*H200 GPUs (each with 96 GiB GPU memory).
- __Parameter for sbatch/salloc__:
    - `--partition=accel`
    - `--gpus=N`, `--gpus-per-node=N` or similar, with _N_ being the number of GPUs
- __Other notes__:
    - Accel nodes have $SCRATCH on a shared flash storage file system.



(job_scripts_olivia_accel)=
##### Accel job script
_Accel_ jobs are specified just like *normal* jobs except that they also have
to specify `--partition=accel` and the number of GPUs to use.  You can also
specify how the GPUs should be distributed across nodes and tasks.  The
simplest way to do that is, with `--gpus=N` or `--gpus-per-node=N`, where `N`
is the number of GPUs to use.

For a job simple job running one process and using one H200 GPU, the
following example is enough:

    #SBATCH --account=nnXXXXk     # Your project
    #SBATCH --job-name=MyJob      # Help you identify your jobs
    #SBATCH --partition=accel
    #SBATCH --gpus=1              # Total number of GPUs (incl. all memory of that GPU)
    #SBATCH --time=1-0:0:0        # Total maximum runtime. In this case 1 day
    #SBATCH --cpus-per-task=72    # All CPU cores of one Grace-Hopper card
    #SBATCH --mem-per-gpu=100G    # Amount of CPU memory
    #SBATCH --network=single_node_vni # Currently required for OpenMPI to work with single compute node jobs.

There are other GPU related specifications that can be used, and that
parallel some of the CPU related specifications.  The most useful are
probably:

- `--gpus-per-node` How many GPUs the job should have on each node.
- `--gpus-per-task` How many GPUs the job should have per task.
  Requires the use of `--ntasks` or `--gpus`.
- `--mem-per-gpu` How much (CPU) RAM the job should have for each GPU.
  Can be used *instead of* `--mem-per-cpu`, (but cannot be used
  *together with* it).

See [sbatch](https://slurm.schedmd.com/sbatch.html) or `man sbatch`
for the details, and other GPU related specifications.

### Interactive jobs
For technical reasons, interactive jobs started with `salloc` will for
the time being *not* start a shell on the first allocated compute node
of the job.  Instead, a shell is started on the *login node* where you
ran `salloc`.  (This will be changed before the pilot period is over.)

This means that the commands are run on the login node, not in the
compute node.  To run a command on the compute nodes in the
interactive job, either use `srun <command>` or `ssh` into the node
and run it there.

(olivia_internet_proxies)=
## Internet access from compute nodes
While the login nodes are directly connected to the public internet, the compute nodes are not.
To access external resources, for example software repositories, you can use proxies that we provide.
```
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
```
Please be aware that this works only for http/https resources.
For example to access a GitHub repo, you have to use HTTPS and not SSH.


### Performance analysis tools


## Storage
Home directories and project directories are found in the usual place:
`/cluster/home` and `/cluster/projects`.  These have disk quotas; see
`dusage` for usage.

On Olivia, there is no personal work areas
(`/cluster/work/users/<uname>`).  Instead, there are *project* work
areas, one for each project, in `/cluster/work/projects/`.  This makes
it easier to share temporary files within the projects.  There is no
disk quotas in the project work areas.  Later, old files will be
deleted automatically, just like in the user work areas on the other
clusters.


<!--
## Debugging
## Data Storage/access(NIRD) and Transfer
## Guides to use Olivia effectively
### Best Practices on Olivia
## Any monitoring tools for users?
## User stories
## Troubleshooting
## Relevant tutorials
-->


