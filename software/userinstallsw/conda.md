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
You can then share a list of the installed packages with collaborators or
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

We typically provide modules for **Anaconda**, **Miniconda** and **Mamba**.
Either of the three is fine.

The Conda workflow consists of several steps, these are:
- Create an environment
- Source into the environment
- Install software, libraries and/or packages in the environment
- Run software and packages that you have installed.
These steps are performed slightly differently in the clusters than what you 
might do in your local machine.

In the next sections we will go through the different steps on our machines and
provide some useful tips for reproducibility or ease of use.


# Quickstart simple environment setup

In order to create an environment we start by loading Conda module of choice,
this can be either `Anaconda3`, `Miniconda3` or `Mamba`, here we give an example using
the `Anaconda3/2022.10` module, but any other Conda module works
```bash
module load Anaconda3/2022.10
source ${EBROOTANACONDA3}/bin/activate
```
with `Miniconda3` the last command would be
```bash
source ${EBROOTMINICONDA3}/bin/activate
``` 
and with `Mamba`it would be
```bash
source ${EBROOTMAMBA}/bin/activate
``` 

```{note}
The rest of the commands will be the same for all three distributions with 
the exception of that if you are using Mamba you have to change the `conda` 
in all the terminal commands with `mamba`, e.g. `mamba install numpy`.
```

At this point you are should be in the `base` environment of Conda (your 
terminal should have `(base)` before the directory name), where you can run
basic Python scripts, use the Conda executable to create and manage 
environments, install software or packages, etc.

Conda downloads and stores a lot of files when installing packages in 
environments. Conda stores files in a cache to speed up subsequent
installations of the same software. This can lead to cluttering of your home 
directory if not careful. To avoid cluttering your home directory, we encourage
you to specify a path to where you want to create the environment and where you
want to keep the software cache. A good location is your project directory for
both the environment and the conda software cache.

To specify the software cache you can run this command in your terminal
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

When creating the environment we specify the environment path by using the 
`--prefix` option.
```shell
conda create --prefix /cluster/projects/nn____k/conda/my-env 
```
all files related to the environment will now be stored under the
`/cluster/projects/nn____k/conda/my-env/` directory.
To activate this environment we need to specify its path
```shell
conda activate /cluster/projects/nn____k/conda/my-env
```

Notice that the `(base)` in your terminal now has changed to something similar 
to `(my-env)`, this is indication that you are now in the context of the 
`my-env` environment.

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

```{note}
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

```{warning}
Conda will at one point advice you to run the `conda init` command. 
**do not run this command** 
This will change your `.bashrc` and will make it very difficult for support to 
troubleshoot any of your issues.
```

# Using `environment.yml` files

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

If you change the `environment.yml` file for an existing environment, and you want to install the new packages, you run this in your terminal
```shell
conda update --file environment.yml
```

# Activating the environment in your job script

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

# Help! I ran `conda init` and/or I see `(base)` everytime I log in
We advice against running `conda init`, but if you have run it you can "undo" 
this by deleting the lines that Conda added to your `.bashrc` or similar file and restarting your terminal (log off and back in again).

You can find your `.bashrc` file in your home directory. The lines that you must remove are
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cluster/software/Anaconda3/2022.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/software/Anaconda3/2022.10/etc/profile.d/conda.sh" ]; then
        . "/cluster/software/Anaconda3/2022.10/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/software/Anaconda3/2022.10/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
Once you have deleted these lines and restarted your session you will be free of the `(base)` prefix.

# Container solution

If you are interested using Conda through a Singularity/Apptainer container,
have a look at <https://github.com/bast/singularity-conda/>.
