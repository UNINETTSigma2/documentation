(olivia-software )=

# Installing and Using Software on Olivia

```{contents} __Page Overview__
:depth: 3 
```

Olivia is a quite different machine compared to our other HPC clusters.
It built by HPE Cray computer, which provides the Cray Programming Environment (CPE) as means for advanced users to get the optimal performance when self-compiling code.
In contrast to our previous machines, Olivia has a rather large accelerator partition using Nvidia Grace-Hopper cards. These consists on the one hand of the Hopper 200 GPUs but also of the ARM based Grace CPUs, see [here](olivia).

We therefore decided to against trying to provide a single solution but instead offer three distinct methods of installing and loading software targeted at different user groups:
1. Module system providing preinstalled software and libraries.
2. HPC-container-wrapper for python and R users which replaces direct installation using Conda or pip.
3. Containers especially for PyTorch and other AI workflows.


```{danger}
**Python, Miniconda and Ananconda User**

Python and conda installation put a lot of stress and load onto the file system.
To prevent file system slowdowns, __we don't allow native (directly with pip or conda) installations__.
Instead create containerized installations using the HPC-container-wrapper or use apptainer to run containers directly.

Please read the sections about [HPC-container-wrapper](olivia-python) and [AI workflows](olivia-ai) thouroughly.

If you have any questions or need help, please [contact us](support-line).
```

## 1. Module System

The module system provides a convenient way to access software packages and libraries that are installed and maintained specifically for the Olivia HPC cluster.
All software is compiled and optimized for Olivia's nodes to ensure the best performance.

### List Available Stacks

After logging in, no software stack is already loaded, but you easily list the available stacks with:

```bash
$ module available
```

You should see output similar to this:

```
----------------------------- /cluster/software/modules/Core -----------------------------
   CrayEnv (S)    EESSI/2023.06    NRIS/CPU (S)    NRIS/GPU (S)    NRIS/Login (S,D)    init-NRIS (S,L)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
   L:  Module is loaded
   D:  Default Module

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the
"keys".

```

There are three types of stacks installed on Olivia:
1. NRIS software stacks providing software and libraries installed by us
2. EESSI software stack providing software and libraries curated by the European EESSI project
3. Cray programming environment providing compilers, libraries and tooling optimized for HPE Cray systems

###  NRIS Software Stacks

The `NRIS` modules provide preinstalled software tailored for different node types:

- **`NRIS/CPU`**: Contains software packages and libraries optimized for the **CPU compute nodes**.
- **`NRIS/GPU`**: Currently includes libraries, compilers, and tools for building software for the Grace-Hopper 200 GPUs.
  In the future, this stack will also include AI frameworks such as PyTorch and TensorFlow (see _AI Frameworks_ below).
- **`NRIS/Login`**: Includes tools for pre- and post-processing data. 
  **Note**: Do not use this stack for running workflows on the compute nodes.

To load a stack and view its available software, use:

```bash
$ module load NRIS/CPU
$ module avail
```

#### Searching for Modules Across Stacks

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

### Using the EESSI Stack

The EESSI (European Environment for Scientific Software Infrastructure) software stack - optimized for each supported CPU architecture - already available on Betzy, Fram, and Saga is now also available on Olivia.

To load the EESSI stack, simply use:

```bash
$ module load EESSI/2023.06
```

Or:

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
`module load EESSI-extend` or without EasyBuild through loading one of the available `buildenv/*` modules, for example, `module load buildenv/default-foss-2023a`
For more information see the official [EESSI documentation](https://www.EESSI.do/docs/using_EESSI/building_on_EESSI/).

### GPU-enabled Software on EESSI

The official EESSI stack contains already some modules of popular software like GROMACS, but many are also still missing.

To get you started more quickly, we have added some GPU-enabled software on Olivia which are not yet provided by [EESSI](https://www.eessi.io/docs) or supported by [EasyBuild](https://easybuild.io/).

**Note**: If your job requires the use of MPI, please ensure you use `srun` for proper integration.
Sample job script using `OSU-Micro-Benchmarks/7.2-gompi-2023b` from the EESSI stack:


```bash
#!/bin/bash -e
#SBATCH --job-name=Sample
#SBATCH --account=nnXXXX
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G

module load EESSI/2023.06
module load OSU-Micro-Benchmarks/7.2-gompi-2023b

srun osu_bw
```

(olivia-python)=
## 2. Python, R, and (Ana-)Conda

Python and R are widely used in scientific computing, but they were originally designed for personal computers rather than
 high-performance computing (HPC) environments. These languages often involve installations with a large number of small 
files—Python environments, for example, can easily consist of tens or even hundreds of thousands of files. This can strain
 the file system, leading to poor performance and a sluggish user experience.

To address this, Olivia uses the [_HPC-container-wrapper_](https://github.com/CSCfi/hpc-container-wrapper/), a tool designed to encapsulate
 installations within containers optimized for HPC systems. This tool wraps Python or Conda environments inside containers, 
significantly reducing the number of files visible to the file system. It also generates executables, allowing you to run commands 
like `python` seamlessly, without needing to interact directly with the container. This approach minimizes file system load while 
maintaining ease of use.

### Key Features of HPC-container-wrapper

The HPC-container-wrapper supports wrapping:

- **Conda installations**: Based on a [Conda environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).
- **Pip installations**: Based on a [pip requirements.txt file](https://pip.pypa.io/en/latest/reference/requirements-file-format/).
- **Existing installations on the file system**: To reduce I/O load and improve startup times.
- **Existing Singularity/Apptainer containers**: To hide the need for using the container runtime from the user.

### Creating a New Python or Conda Environment

To create a new Python or Conda environment on Olivia, follow these steps:

1. **Load the necessary modules and activate internet access from the compute nodes**:

   ```bash
   $ export http_proxy=http://10.63.2.48:3128/
   $ export https_proxy=http://10.63.2.48:3128/
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

     The `--slim` argument uses a pre-built minimal Python container with a newer Python version as a base. 
Without `--slim`, the host system is fully available, but this may include unnecessary system installations.

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

### Modifying an Existing Environment

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

### Tips and Troubleshooting

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

### Example Workflow: End-to-End Conda Installation

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
   $ module load NRIS/CPU hpc-container-wrapper
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

For more examples and advanced usage, see the [HPC-container-wrapper GitHub repository](https://github.com/CSCfi/hpc-container-wrapper) and check the [documentation page of CSC about HPC-container-wrapper](https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#wrapping-a-plain-pip-installation).

(olivia-ai)=
## 3. AI Frameworks

For AI workflows based on popular frameworks like PyTorch, JAX, or TensorFlow, we aim to deliver optimal performance by utilizing containers provided by [NVIDIA](https://catalog.ngc.nvidia.com/containers). These containers are specifically optimized for GPU workloads, ensuring excellent performance while remaining relatively straightforward to use.

```{danger}
Please __do not install PyTorch directly via pip or conda__ as this puts a lot of stress on the file system.

Instead use the containers presented below or the module we will provide soon (documentation on that will follow).
```

### Downloading Containers

You can use pre-built containers or download and convert your own. Here are the options:

1. **Pre-Built Containers**:
   Some NVIDIA containers are already available in the Apptainer image format (`.sif`) under:
   `/cluster/work/support/container`.

2. **Downloading Your Own Containers**:

   You can download containers from sources like the [NVIDIA NCC catalogue](https://catalog.ngc.nvidia.com/containers) or [Docker Hub](https://hub.docker.com/). Use the following command to download and convert a container into the Apptainer format:

   ```bash
   $ apptainer pull --arch ARCHITECTURE --disable-cache docker://<docker_image_url>
   ```
   By default this command downloads the container for the CPU architecture of the nodes you are running on. That means that you have to specify `--arch arm64` if you download the container from the login nodes but want to run the container on the accelerator/GPU nodes.

   For example, to download and convert the latest NVIDIA PyTorch container to run on the GPU nodes (that use ARM CPUs):

   ```bash
   $ apptainer pull --arch arm64 --disable-cache docker://nvcr.io/nvidia/pytorch:25.09-py3
   ```

   For more options, check the Apptainer build help:

   ```bash
   $ apptainer pull --help
   ```

#### Important Notes
- **Building Your Own Containers:**
  Currently it is only possible to build containers from definition files that don't require root privileges inside the container. We are working on enabling some version of fakeroot to enable most builds in the future.
- **Running on GPU Nodes**:
  If you plan to run these containers on GPU nodes, ensure that you download the container from a GPU node. This can be done either via a job script or an interactive job. This ensures that the container is built for the correct architecture (e.g., ARM64).
  For details on accessing internet resources from compute nodes, see [this section](olivia_internet_proxies).

- **Managing Cache**:
  By default, Apptainer stores its cache in your home directory, which may lead to `disk quota exceeded` errors. To avoid this, use `--disable-cache* or set the cache directory to your project work area:

  ```bash
  $ export APPTAINER_CACHEDIR=/cluster/work/projects/<project_number>/singularity
  ```

### Running Containers

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

4. **Job Script Setup for multi GPU (single node/multi node)**:

    While experimenting, we encountered cases where the `torchrun` was not recognized unless its full path was explicitly specified. We can set the path to the `torchrun` as `TORCHRUN_PATH="/usr/local/bin/torchrun`, then bind it with the apptainer exec command as `apptainer exec --nv --bind  $TORCHRUN_PATH ./your_script.py`.

    Moreover , if we need GPUs across multiple nodes, we have to take into consideration that some of the framework like Libfabric might not be installed on the container. So, we need to explicitly bind it so that our container could be use to run on multiple nodes.

    In our host system, we identified that the libfabric is available at this location `/opt/cray/libfabric/1.22.0/lib64` . The code written below on the job script will add `--bind /opt/cray/libfabric/1.22.0/lib64:/usr/lib64` to the apptainer command which ensures that the libfabric libraries from the host system are available inside the container at `/usr/lib64``

    ````bash
    # Bind libfabric (adjust the path based on your host system)
    LIBFABRIC_PATH="/opt/cray/libfabric/1.22.0/lib64"

    # Explicitly specify the full path to torchrun
    TORCHRUN_PATH="/usr/local/bin/torchrun"

    # Run the training script with torchrun inside the container
    srun apptainer exec --nv --bind  $LIBFABRIC_PATH:/usr/lib64  $TORCHRUN_PATH --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$RANDOM .....
    ````

### Enabling GPU Support

To enable GPU support inside the container, add the `--nv` flag to your Apptainer command. This ensures that the container has access to the GPU resources. For example:

```bash
$ apptainer exec --nv /cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

This command checks if GPUs are available and prints the number of GPUs detected inside the container.


### PyTorch Example Workflow

We have provided a comprehensive example on how to run PyTorch in a container on [this page](pytorch-olivia).


### Next Steps

Currently, you need to run these containers directly using Apptainer. However, we are working on simplifying the experience by integrating these tools into the module system. This will allow you to:

- Load a module that provides executables like `python` and `torch`, eliminating the need to interact with the containers directly.
- Seamlessly use AI frameworks without worrying about container management.
