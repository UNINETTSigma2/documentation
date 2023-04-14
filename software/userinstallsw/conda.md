(installing-with-conda)=
# Installing software with Conda (Anaconda & Miniconda)

You can install many R, Python, and other packages yourself using
[Conda](https://docs.conda.io/en/latest/). Conda enables you to easily install complex packages and software. Creating multiple environments allows you to have installations of the same software in different versions or incompatible software collections at once. You can easily share a list of the installed packages with collaborators or colleagues, so they can set up the same environment in a matter of minutes.

Anaconda and Miniconda are two distribution options for Conda. Anaconda is a full-fledged distribution that includes a large number of pre-installed packages, while Miniconda is a minimal distribution that comes with only the essential packages required to set up Conda. If you prefer a lightweight installation and want more control over the packages you install, Miniconda might be a better choice. On the other hand, if you want a comprehensive set of packages out of the box, Anaconda is the way to go.


## Setup

First, you need to load the Miniconda or Anaconda module, which serves as a package manager for Python and R. Conda makes it easy to have multiple environments, allowing you to run different versions of the same software or incompatible software collections without interference.

### Load conda module
Start by removing all preloaded modules, which can complicate things. We then display all installed versions and load the newest Miniconda or Anaconda:

  ``` sh
  $ ml purge
  $ ml avail conda
  ```

We then load Miniconda or Anaconda:

  ``` sh
  $ ml Miniconda3/4.9.24 # Replace with one of the versions available on the system
  ```
  or
  ``` sh
  $ ml Anaconda3/2022.05 # Replace with one of the versions available on the system
  ```
### Setup conda activate command

If this is your first time using Conda on the system or you haven't initialized Conda for your shell before, you need to initialize it. This enables you to use conda activate. The shell is initialized as follows:

``` sh
$ conda init bash
$ source ~/.bashrc
```
Here, we use bash as an example. If you are using a different shell, replace it with the appropriate shell name.

### Creating the conda environment
It is possible to create a Conda environment in your home directory like you would on a local machine. See the official [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on managing environments.  However, this is not recommended on our HPC systems, as each user's storage is limited, and Conda environments can take up a lot of space. Instead, we recommend creating a Conda environment in a directory under your project directory. This also has the advantage of making the environment accessible to others in your project, which can facilitate collaboration.

Specifying the Python version when creating the environment ensures compatibility, reproducibility, and optimal performance. It helps to avoid potential conflicts, makes debugging easier, and simplifies collaboration.

First make a directory for your conda environment:

``` sh
$ mkdir -p /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT
```
Replace `nnXXXXk` with your own project identifier, and `PATH_TO_ENVIRONMENT` with the path to where you wish to store the environment.
Then, create the environment

``` sh
conda create -p /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT python=3.10
```
Replace `ENVIRONMENT` with a name for your Conda environment. Make sure to specify the desired Python version to ensure compatibility, reproducibility, and optimal performance for your project.

### Installing packages
Once you have created your Conda environment, you can install packages using the conda install command. You can also specify the channels to search for packages, such as `conda-forge` and `bioconda`.

Before installing packages, activate your Conda environment:

``` sh
$ conda activate /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME
```

To install a package, simply use the `conda install` command followed by the package name:

``` sh
$ conda install PACKAGE_NAME
```
Replace `PACKAGE_NAME` with the name of the package you wish to install.

To install a package from a specific channel, such as `conda-forge` or `bioconda`, use the `-c` option followed by the channel name:
``` sh
$ conda install -c conda-forge PACKAGE_NAME
```
Replace PACKAGE_NAME with the name of the package you wish to install from the conda-forge channel.

For bioconda, the command would look like:
``` sh
$ conda install -c bioconda PACKAGE_NAME
```
Replace PACKAGE_NAME with the name of the package you wish to install from the bioconda channel.

You can also specify multiple channels by including multiple -c options:
``` sh
$ conda install -c conda-forge -c bioconda PACKAGE_NAME
```
Remember to replace `PACKAGE_NAME` with the name of the package you wish to install.

### Creating a Conda environment from an environment.yml file
An `environment.yml` file is a configuration file that specifies the packages, channels, and Python version required for a Conda environment. Creating a Conda environment from an `environment.yml` file ensures that your environment is reproducible and consistent across different systems.

First, make sure you have an `environment.yml` file. Here's an example of what it might look like:
``` yaml
name: ENVIRONMENT_NAME
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - PACKAGE_NAME_1
  - PACKAGE_NAME_2
  - PACKAGE_NAME_3
```

Replace ENVIRONMENT_NAME with the name of your Conda environment, and `PACKAGE_NAME_X` with the names of the packages you wish to install.

To create the environment using the `environment.yml` file, run the following command:

  ``` sh
  $ conda env create -f environment.yml -p /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME
```
Replace `nnXXXXk` with your own project identifier, `PATH_TO_ENVIRONMENT` with the path to where you wish to store the environment, and `ENVIRONMENT_NAME` with the name of your Conda environment.

This command will create a new Conda environment at the specified path, using the packages, channels, and Python version defined in the `environment.yml` file. Once the environment is created, you can activate it using:

``` sh
$ conda activate /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME
```
Now your Conda environment is set up and ready to use with the packages specified in the `environment.yml` file.

### Exporting the Conda environment
Exporting a Conda environment allows you to share the environment configuration with others, making it easy for collaborators to recreate the same environment on their systems. To export a Conda environment, you can use the conda env export command, which generates an `environment.yml` file containing the environment's channels, packages, and Python version.

``` sh
$ conda activate /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME
```
Replace `nnXXXXk` with your own project identifier `PATH_TO_ENVIRONMENT` with the path to where you stored the environment, and `ENVIRONMENT_NAME` with the name of your Conda environment.

Next, use the `conda env export` command to export the environment:

``` sh
  $ conda env export > environment.yml
```
This command will create an `environment.yml` file in the current directory containing the environment's configuration. You can now share this file with collaborators or save it for future use.

If you want to export the environment file to a specific location or with a different name, provide the full path and filename after the `>` operator:

``` sh
$ conda env export > /path/to/directory/your_filename.yml
```
Replace `/path/to/directory/your_filename.yml` with the desired path and filename for the exported environment file.

By sharing the exported `environment.yml` file with your collaborators, they can easily recreate the same Conda environment on their systems, ensuring consistent package versions and dependencies.


### Installing packages with pip in a Conda environment
While Conda is the preferred package manager for packages in a Conda environment, you might encounter some packages that are not available through Conda channels. In such cases, you can use pip to install the package.

First, make sure your Conda environment is activated:

``` sh
$ conda activate /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME
```
Replace `nnXXXXk` with your own project identifier, `PATH_TO_ENVIRONMENT` with the path to where you stored the environment, and `ENVIRONMENT_NAME` with the name of your Conda environment.
Next, install the package using pip:

``` sh
$ pip install PACKAGE_NAME
```
Replace `PACKAGE_NAME` with the name of the package you want to install.

Although mixing Conda and pip installations is generally not recommended, sometimes it is unavoidable. When using pip in a Conda environment, it's important to be aware of potential dependency conflicts and to resolve them if necessary.

You should first try to find the package in Conda channels, like `conda-forge` or `bioconda`, before resorting to pip. By using Conda channels, you can take advantage of Conda's dependency management features and avoid potential conflicts.

If you need to use pip, install the package within your Conda environment, as shown above, to keep your environment self-contained and maintain proper dependency management.

### Activating a Conda environment in a Slurm script

When submitting a job using a Slurm script, you'll need to activate your Conda environment within the script. To do this, simply include the appropriate conda activate command in your Slurm script.

Here's an example Slurm script that demonstrates how to activate a Conda environment:

  ``` sh
# SBATCH directives

# Load the Conda module
ml purge
ml Miniconda3/4.9.24 # Replace with the version available on the system

# Activate the Conda environment
conda activate /cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME

# Run your code
python my_script.py
```
In this example, replace `/cluster/projects/nnXXXXk/PATH_TO_ENVIRONMENT/ENVIRONMENT_NAME` with the correct path to your Conda environment.

Remember to load the Conda module and activate the environment before running any code that depends on the packages in your Conda environment. This will ensure that your job has access to the correct package versions and dependencies.

### Common Questions and Pitfalls
#### Disk Quota Exceeded error message
Conda environments contain a lot of files, which can cause you to exceed your number of files quota. This happens especially easily when installing Conda environments in your home folder. Check your quota with dusage.

To solve this error and reduce your number of files, delete unnecessary and cached files with:

  ``` sh
  $ conda clean -a
```
To avoid this error, create your Conda environments in your project folder by using the `--prefix PATH` option, as described earlier in this guide.

#### Suppressing the warning about a newer version of Conda

To suppress the warning that a newer version of Conda exists, which is usually not important for most users and will be fixed by us by installing a new module, run the following command:

``` sh
$ conda config --set notify_outdated_conda false
```

This setting will prevent Conda from displaying a warning message about outdated Conda versions, making your Conda usage experience smoother.