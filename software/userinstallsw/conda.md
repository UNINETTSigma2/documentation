(installing-with-conda)=

# Installing software with Conda (Anaconda & Miniconda)

```{contents} Table of Contents
```

[Conda](https://docs.conda.io/en/latest/) is a **tool to install packages**
(Python/R/C/C++/...) and their dependencies.  You can do this yourself
without administrator permissions on the cluster.

It is also a tool that allows to **create and keep track of different
environments** for different projects.
Creating multiple environments allows you to have installations of the same
software in different versions or incompatible software collections at once.
You can then easily share a list of the installed packages with collaborators or
colleagues, so they can set up the same environment in a matter of minutes.

Conda comes in different shapes and forms but the idea and functionality is
more or less the same:
- [Anaconda](https://www.anaconda.com/): Full-fledged distribution that
  includes a large number of pre-installed packages. Use this if you don't want
  to install any packages and just need to test something and are sure that the
  package you need is part of Anaconda.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Minimal
  distribution that comes with only the essential packages required to set up
  Conda. Use this if you want to install dependencies and keep track of your
  environment.
- [Mamba](https://mamba.readthedocs.io/): A re-implementation of Conda for
  fast dependency resolution.
- [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
  Ultra-lightweight Mamba which supports the most important `conda` commands.

We typically provide modules for **Anaconda** and **Miniconda**. Either of the
two is fine.

The Conda workflow consists of several steps, these are:
- Create an environment
- Source into the environment
- Install software, libraries and/or packages in the environment
- Run software and packages that you have installed.
These steps are performed slightly differently in the clusters than what you 
might do in your local machine.

In the next sections we will go through the different steps on our machines, 
provide some useful tips for reproducibility or ease of use and finally
showcase a couple of common pitfalls to avoid. 


# Quickstart simple environment setup

Following is a simple setup which can be used when testing, debugging or any 
lightweight applications. More general usage will be shown in the following 
sections. 

In order to create an environment we start by loading the desired Conda module,
this can be either `Anaconda3` or `Miniconda3`, here we give an example using
the `Anaconda3/2022.05` module, but any other Conda module works
```bash
module load Anaconda3/2022.05
source ${EBROOTANACONDA3}/bin/activate
```
with `Miniconda3` the last command would be
```bash
source ${EBROOTMINICONDA3}/bin/activate
``` 
This should make it so you are in the `base` environment of Conda (your 
terminal should have `(base)` before the directory name), where you can run
basic Python scripts, use the Conda executable to create and manage 
environments, install software or packages, etc.


Creating an environment is straight forward, run the following command to 
create an environment named `my-env`
```shell
conda create --name my-env
```
This will create an empty environment which you can activate
```shell
conda activate my-env
```
Notice that the `(base)` in your terminal now has changed to `(my-env)`, this
is indication that you are now in the context of the `my-env` environment.


Once you have **activated** the environment you can install software or libraries, `numpy` 
for example, by running
```shell
conda install numpy
```
Now any python script that utilizes numpy can be run in the terminal using the
version of numpy you have installed in the step above.

Some of the libraries or software you install might need to be installed from 
different channels than the default ones. Scipy, for example, can be installed from 
the channel `conda-forge`. This can be done in a number of ways, following we introduce some

```shell
conda install conda-forge::scipy
```
alternatively
```shell
conda install --channel conda-forge scipy
```
both will do the same operation.

Some packages can only be installed with `pip`, but this can also be done through Conda:
```shell
conda install pip
pip install jupyter
```
These two commands will first install `pip` in the context of the Conda 
environment (otherwise you would use the global installation, which can 
introduce other problems) and then use this `pip` to install the package 
`jupyter`.

```{'note'}
`pip` might not need to be installed this way all the time, as it might have been 
installed by a previous command (such as installing any Python version). This 
is still good practice to avoid different versions of the same packages used 
simultaneously.
```

You can at any time list the packages installed in the environment by running
```shell
conda list
```
which, for the packages we have installed here, will show something like this:
```shell
# packages in environment at /path/to/my-env:
#
# Name                    Version                   Build  Channel
.
.
.
jupyter                   1.0.0                    pypi_0    pypi
numpy                     2.0.0           py312h8813227_0    conda-forge
pip                       24.0               pyhd8ed1ab_0    conda-forge
python                    3.12.4          h37a9e06_0_cpython    conda-forge
scipy                     1.14.0          py312hb9702fa_1    conda-forge
.
.
.
```
where we have omitted most of the other dependencies.

Once you are done with your environment you can delete it with
```shell
conda deactivate
conda remove --name my-env --all
```

```{'warning'}
Conda will at one point advice you to run the `conda init` command. 
**do not run this command** 
This will change your .bashrc and will make it very difficult for support to 
troubleshoot any of your issues.
```

```{'note'}
The sequence of commands above will create the Conda environment and install 
packages in the default Conda location. This location is in you `home` 
directory. Due to this, cluttering might arise when working with many and/or 
bigger environments. The following sections will show best practices when 
scaling up your usage of environments.
```

# Best practices for large scale uses

As noted above, for more heavyweight uses certain precautions need to be taken.
Additionally one would like to be able to share environments with collaborators
and be able to run the software installed through the slurm queue system, the
next sections outline how to do all this.

## Specify environment and software cache directory

There is two main problems that need to be addressed, software chache storage and 
environment location. These two can be addressed at the same time by specifying
a directory where to build the environment and store the software cache.

In this example we use a generic project directory `nn____k` which has to be 
changed to your actual project.
After sourcing into `base` (see above steps)
we can export the `CONDA_PKGS_DIRS` variable to something that fits our 
purposes.
```shell
export CONDA_PKGS_DIRS=/cluster/projects/nn____k/conda/package-cache
```
The `package-cache` stores tar-balls, logfiles and other side products of 
software installation. Some of these files are stored to make subsequent 
installations in different environments more streamlined.
This cache can be cleaned by running 
```shell
conda clean -a
```
which will remove these files. THis is specially useful if you have created 
smaller environments in your home directory and need to clean up.

When creating the environment we specify the environment path by using the 
`--prefix` option.
```shell
conda create --prefix /cluster/projects/nn____k/conda/my-env 
```
all binaries will now be stored under the `/cluster/projects/nn____k/conda/my-env/`.
To activate this environment we need to specify its path
```shell
conda activate /cluster/projects/nn____k/conda/my-env
```

## Using `environment.yml` files

One of the pros of using Conda is that you can share your environment 
specification to collaborators through `environment.yml` files, which they can
use to create their own copy of your environment.

Such an `environment.yml`file can look like this:
```yaml
name: my-env
channels:
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
```
In order to create an environment from this file you run
```shell
conda create --prefix /cluster/projects/nn____k/conda/my-env --file environment.yml
```
given that the file is named `environment.yml`.
This will install all the dependencies listed from the necessary channels. 

To create this file you can either write the `environment.yml` file manually following [these instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)
or use conda to export the list of installed packages in your environment 
automatically. You can do this by first activating the environment 
`conda activate /path/to/my-env` and then run this command
```shell
conda env export > environment.yml
```
This will overwrite any `environment.yml` file in the current directory and 
list all installed packages in the environment. These can be quite many, due to 
system specific dependencies that are installed  at the same time. This might 
make this environment file not compatible across platforms. In order to just 
show the packages you specifically asked for you can run the following command
```shell
conda env export --from-history
```


## Activating the environment in your job script

We activate the environment in the job script the same way we activate it
interactively on the command line (above). The additional `SBATCH`
directives on top are unrelated to the Conda part:
```{code-block} bash
---
emphasize-lines: 16-17, 22
linenos:
---
#!/usr/bin/env bash

# settings to catch errors in bash scripts
set -euf -o pipefail

#                change this
#                    |
#                    v
#SBATCH --account=nn____k
#SBATCH --job-name=example
#SBATCH --qos=devel
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

# the actual module version might be different
module load Anaconda3/2022.10
source ${EBROOTANACONDA3}/bin/activate

#                               change this
#                                   |
#                                   v
conda activate /cluster/projects/nn____k/conda/myproject

python --version
python example.py
```

We need three lines before running any code that depends on the packages in
your environment: loading the module, sourcing the activate script, and `conda
activate` your environment.

If you used Miniconda instead of Anaconda, then lines 16 and 17 (above) might
look like this instead (version might be different):
```bash
module load Miniconda3/22.11.1-1
source ${EBROOTMINICONDA3}/bin/activate
```

## How to speed up the installation with [Mamba](https://mamba.readthedocs.io/)
Instead of `conda env create`, you can use `mamba env create` to speed up the
installation.  Mamba is a re-implementation of Conda for fast dependency
resolution. We have modules for Mamba on all our clusters. Try: `module avail mamba`.


## Container solution

If you are interested using Conda through a Singularity/Apptainer container,
have a look at <https://github.com/bast/singularity-conda/>.
