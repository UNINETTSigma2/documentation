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
where we have ommitted most of the other dependencies.

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
which will remove these files. 

When creating the environment we specify the environment path by using the 
`--prefix` option.
```shell
conda create --prefix /cluster/projects/nn____k/conda/my-env 
```
all binaries will now be stored under the `/cluster/projects/nn____k/conda/my-env/`.

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
`conda activate my-env` and then run this command
```shell
conda env export > environment.yml
```
This will overwrite any `environment.yml` file in the current directory and 
list all installed packages in the environment. These can be quite many, due to 
system specific dependencies that are installed  at the same time. This might 
make this environment file not compatible accross platforms. In order to just 
show the packages you specifically asked for you can run the following command
```shell
conda env export --from-history
```



### Do not install into your home directory otherwise you fill your disk quota

**Conda environments can contain hundreds of thousands of files**, which can cause
you to exceed your number of files quota.  **Place both the environment and the
package cache outside of your home directory** (see examples on this page).

If you accidentally install into your home and fill up your quota, first
inspect with:
```console
$ dusage
```
See also our documentation on {ref}`storage-quota`.

If this happens, you can try to delete unnecessary and cached files with:
```console
$ conda clean -a
```

Another solution (but please do this carefully) is to remove everything in
`~/.conda` except `~/.conda/environments.txt`.

Normally the only two Conda files that need to be in your home directory are `~/.conda/environments.txt`
and possibly `~/.condarc`. Everything else should be somewhere else: project folder
or work area.



## Installing packages/environment from an `environment.yml` file

We recommend to always start from an `environment.yml` file to create an
environment for your project and to install packages into it.  Further down we
also explain how to create such a file if you don't have one yet.

Here is an example `environment.yml` file:
```yaml
name: myproject
channels:
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
```

The file describes the environment with the name "myproject" (better change
that to something more descriptive!), which dependencies it needs and from
which "channels" to install from. If you want, you can specify precise versions
for each dependencies. If you leave them out, it will try to install the latest
versions which match other specifications in that file.

Should one specify versions or not?
- When a project is in progress and evolving, it is often better to leave them out
  and use the latest versions.
- However, when a project is completing and about to be published or shared with
  collaborators, it may be better to specify versions.

Once you have such a file you have achieved two things:
- You have documented your dependencies (great for reproducibility)
- You can install from it

To install from that file, you only need to decide where to. **Not into your
home directory**, otherwise you will fill your storage quota.

Here I create a Conda environment from an `environment.yml` file with a script but you
can also do that command after command:
```{code-block} bash
---
emphasize-lines: 11
linenos:
---
#!/usr/bin/env bash

# settings to catch errors in bash scripts
set -euf -o pipefail

module load Anaconda3/2022.10

#                                 change this
#                                     |
#                                     v
my_conda_storage=/cluster/projects/nn____k/conda

export CONDA_PKGS_DIRS=${my_conda_storage}/package-cache
conda env create --prefix ${my_conda_storage}/myproject --file environment.yml
```

```{admonition} How to speed up the installation with [Mamba](https://mamba.readthedocs.io/)
Instead of `conda env create`, you can use `mamba env create` to speed up the
installation.  Mamba is a re-implementation of Conda for fast dependency
resolution. We have modules for Mamba on all our clusters. Try: `module avail mamba`.
```

You need to adapt the location (line 11) and also change the name ("myproject").
On line 13 we define `CONDA_PKGS_DIRS` to also be in your well-defined `my_conda_storage`,
otherwise the package cache is in your home directory and that is a problem.

If I run this for my example `environment.yml` above it creates the cache and environment folders
and each contains many files already:
```console
$ find . -maxdepth 1 -type d -exec sh -c 'echo -n "{}: "; find "{}" -type f | wc -l' \; | sort -n -k2 -r

.: 31510
./package-cache: 16549
./myproject: 14961
```

35 thousand files! You can easily get up to 100 thousand or more files and that
is too much for your home directory.


## But I don't have an environment.yml file!

Don't worry, you can create one.  Let us look again at our example
`environment.yml` file:
```yaml
name: myproject
channels:
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
```

This is equivalent to installing them one by one into an **active** environment:
```console
$ conda install -c defaults python=3.10
$ conda install -c defaults numpy
$ conda install -c defaults pandas
$ conda install -c defaults scipy
```

Now if somebody asks you to install like this:
```console
$ conda install -c bioconda somepackage
$ conda install -c bioconda otherpackage
```
... then you know what to do and can create an `environment.yml` file from it:
```yaml
name: myproject
channels:
  - bioconda
dependencies:
  - somepackage
  - otherpackage
```

Please also look at the [Conda
documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)
which shows an even simpler `environment.yml`.


## If I change the environment.yml file, do I need to remove everything and restart from scratch?

No need to remove everything. You can adjust your `environment.yml` and all you need to change
is `conda env create` to `conda env update`:
```{code-block} bash
---
emphasize-lines: 14
linenos:
---
#!/usr/bin/env bash

# settings to catch errors in bash scripts
set -euf -o pipefail

module load Anaconda3/2022.10

#                                 change this
#                                     |
#                                     v
my_conda_storage=/cluster/projects/nn____k/conda

export CONDA_PKGS_DIRS=${my_conda_storage}/package-cache
conda env update --prefix ${my_conda_storage}/myproject --file environment.yml
```


## Should I have one or many environment.yml files?

We recommend one **environment per project**, not one for all projects.  Here
meaning research project/ code project, not compute allocation project.  It is
OK to share an environment with colleagues if they use the same code but it is
a good idea to not try to have a single environment for "everything" and all your
many projects.

The reason is that one day you will want to share the environment with somebody
else and the somebody else does not want to install everything to run that one
tiny script that only needs a tiny environment.

Another reason to have one environment per project is that projects can have
different and conflicting dependencies.


## Activating the environment interactively

We need three commands:
```bash
$ module load Anaconda3/2022.10
$ source ${EBROOTANACONDA3}/bin/activate

$ conda activate /cluster/projects/nn____k/conda/myproject
```

If you used Miniconda instead of Anaconda, then the first two lines change:
```bash
$ module load Miniconda3/22.11.1-1
$ source ${EBROOTMINICONDA3}/bin/activate

$ conda activate /cluster/projects/nn____k/conda/myproject
```

Of course you need to adapt `nn____k` to your project and need to change
`myproject` to your actual environment name. Also the versions of the Anaconda
and Miniconda modules evolve - please check with `module avail conda`.

Once activated, you can run your script **inside that environment**. Here I run
a Python script called `example.py`. Notice how the environment in my prompt
changes from nothing to `(base)` to
`(/cluster/projects/nn____k/conda/myproject)`:
```bash
[user@login-1.FRAM ~/example]$ module load Anaconda3/2022.10

[user@login-1.FRAM ~/example]$ source ${EBROOTANACONDA3}/bin/activate

(base) [user@login-1.FRAM ~/example]$ conda activate /cluster/projects/nn____k/conda/myproject

(/cluster/projects/nn____k/conda/myproject) [user@login-1.FRAM ~/example]$ python example.py
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

The `source ${EBROOTMINICONDA3}/bin/activate` line basically replaces a previous
`conda init` which would have modified `.bashrc`.


## Exporting your environment to an environment.yml file

The `environment.yml` file is very precious since it lists all your
dependencies and if you want also their precise versions. It is precious
because your future you and other people will be able to (re)create their
environment from this file and will be able to reproduce your research.
Without this information, they will have a hard time.

Here is an example `environment.yml` file:
```yaml
name: myproject
channels:
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
```

If you read this documentation page and followed our recommendations, then you
never need to export your environment to an `environment.yml` file since you
already have it since you always installed new dependencies from it.

But if you lost the file or installed on the command line and forgot what
packages exactly, you can create it:
```bash
# the first three lines depend on how you created your environment
$ module load Anaconda3/2022.10
$ source ${EBROOTANACONDA3}/bin/activate
$ conda activate /cluster/projects/nn____k/conda/myproject

# this is the interesting line
$ conda env export --from-history > environment.yml
```

If you need precise versions and all dependencies, also dependencies of your
dependencies, you can get more information with:
```
$ conda env export > environment.yml
```


## "WARNING: A newer version of conda exists"

When installing or updating packages you might see this warning:
```
==> WARNING: A newer version of conda exists. <==
  current version: 4.12.0
  latest version: 23.3.1

Please update conda by running

    $ conda update -n base -c defaults conda
```

Do **not** run the command `conda update -n base -c defaults conda` (you do not
have the permissions anyway).  You can almost always ignore this warning but
you can also configure Conda to not bother you with this warning:
```bash
$ conda config --set notify_outdated_conda false
```

This setting is written to your `~/.condarc` if you change your mind and want
to remove this setting later.


## Container solution

If you are interested using Conda through a Singularity/Apptainer container,
have a look at <https://github.com/bast/singularity-conda/>.
