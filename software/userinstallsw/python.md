(installing-python-packages)=

# Installing Python packages

`pip` and `conda` are the easiest ways of installing python packages and
programs as user. In both cases it is advised to use virtual environments to
separate between different workflows/projects. This makes it possible to have
multiple versions of the same package or application without problems of
conflicting dependencies.

## Virtual environments

Virtual environments in Python are a nice way to compartmentalize package
installation. You can have many virtual environment and we recommend that you at
least have one for each disparate experiment. One additional benefit of this
setup is that it allows other researchers to easily replicate your setup.

`pip` is the main package installer for Python and included in every Python
installation. It is easy to use and can be combined with `virtualenv` to manage
independent environments. These can contain different Python versions and packages.

In some cases, packages installed with `pip` have problems with complex dependencies
and libraries. In this case, `conda` is the better solution.

### Setup and installation with pip

Users can install Python packages in a virtual Python environment. Here is how
you create a virtual environment with Python:

``` sh
# First load an appropriate Python module (use 'module avail Python/' to see all)
$ module load Python/3.8.6-GCCcore-10.2.0
# Create the virtual environment.
$ python -m venv my_new_pythonenv
# Activate the environment.
$ source my_new_pythonenv/bin/activate
# Install packages with pip. Here we install pandas.
$ python -m pip install pandas
```

After the analysis is finished the environment can be unloaded or deactivated 
using one of the two methods below. 

1. Close the current terminal 
2. Use the *deactivate* command


In a job script (described below), there is no need to deactivate as the 
environment is only active in the *shell* the job was running in.

For more information, have a look at the [official
`pip`](https://pip.pypa.io/en/stable/) and
[`virtualenv`](https://virtualenv.pypa.io/en/latest/) documentations.

```{note}
When running software from your Python environment in a batch script, it is
_highly_ recommended to activate the environment _only_ in the script (see
below), while keeping the login environment clean when submitting the job,
otherwise the environments can interfere with each other (even if they are the
same).
```

### Using the virtual environment in a batch script

In a batch script you will activate the virtual environment in the same way as
above. You must just load the python module first:

```
# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

# Load modules
module load Python/3.8.6-GCCcore-10.2.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source my_new_pythonenv/bin/activate

# execute example script
python pdexample.py
```

### Sharing package configuration

To allow other researchers to replicate your virtual environment setup it can be
a good idea to "freeze" your packages. This tells `pip` that it should not
silently upgrade packages and also gives a good way to share the exact same
packages between researchers.

To freeze the packages into a list to share with others run:
```bash
$ python -m pip freeze --local > requirements.txt
```

The file `requirements.txt` will now contain the list of packages installed in
your virtual environment with their exact versions. When publishing your
experiments it can be a good idea to share this file which other can install in
their own virtual environments like so:

```bash
$ python -m pip install -r requirements.txt
```

Your virtual environment and the new one installed from the same
`requirements.txt` should now be identical and thus should replicate the
experiment setup as closely as possible.


## Anaconda, Miniconda & Conda

You can install many python and non-python packages yourself using
[conda](https://docs.conda.io/en/latest/) or especially for bioinformatics
software [bioconda](https://bioconda.github.io/).

Conda enables you to easily install complex packages and software.
Creating multiple environments enables you to have installations of the
same software in different versions or incompatible software collections
at once.
You can easily share a list of the installed packages with 
collaborators or colleagues, so they can setup the same
environment in a matter of minutes.

### Setup

First you load the miniconda module which is like a python and r package
manager.
Conda makes it easy to have multiple environments for example one python2 and
one python3 based parallel to each other without interfering.

#### Load conda module
Start by removing all preloaded modules which can complicate things. We then
display all installed version and (on Saga) load the newest Miniconda one (4.6.14):

``` sh
$ ml purge
$ ml avail conda
$ ml Miniconda3/4.6.14
```

On Fram, Miniconda is not installed (yet) but instead you can load `Anaconda3`.


#### Setup conda activate command
To use `conda activate` interactively you have to initialise your shell once with:

``` sh
$ conda init bash
```

#### Add channels
To install packages we first have to add the package repository to conda
(we only have to do this once). This is the place conda will download the packages
from.

```sh
$ conda config --add channels defaults
$ conda config --add channels conda-forge
```

If you want install bioinformatics packages you should also add
the bioconda channel:

``` sh
$ conda config --add channels bioconda
```

#### Suppress unnecessary warnings
To suppress the warning that a newer version of conda exists which is usually
not important for most users and will be fixed by us by installing a new module:

``` sh
$ conda config --set notify_outdated_conda false
```

#### Create new environment

New environments are initialised with the `conda create`. During the creation you
should list all the packages and software that should be installed in this
environment instead of creating an empty one and installing them one by one. This
makes the installation much faster and there is less chance for conda to get stuck in
a dependency loop.

```sh
$ conda create --name ENVIRONMENT python=3 SOMESOFTWARE MORESOFTWARE
```

If you are planning on adding many libraries to your environment, you should
consider placing it in a directory other than your $HOME, due to the
{ref}`storage restrictions <clusters-homedirectory>` on that folder. One
alternative could be to use the {ref}`Project area <project-area>`, please
check out {ref}`Storage areas on HPC clusters <clusters-overview>` for other
alternatives. To install conda in an alternative location, use the `--prefix
PATH` or `-p PATH` option when creating a
new environment.

```sh
conda create -p PATH SOMEPACKAGES
```

This enables multiple users of a project to share the conda environment by
installing it into their project folder instead of the user's home.

### Daily usage

#### Interactively

To load this environment you have to use the following commands either on the
command line or in your job script:

``` sh
$ ml purge
$ ml Miniconda3/4.6.14 # Replace with the version available on the system
$ conda activate ENVIRONMENT
```

Then you can use all software as usual.

To deactivate the current environment:

``` sh
$ conda deactivate
```

If you need to install additional software or packages,
we can search for it with:

``` sh
$ conda search SOMESOFTWARE
```

and install it with:

``` sh
$ conda install -n ENVIRONMENT SOMESOFTWARE
```

or alternatively when creating with a path:

``` sh
$ conda install -p PATH SOMESOFTWARE
```

If the python package you are looking for is not available in conda you can use
[pip](https://pip.pypa.io/en/stable/) like usually from within a conda
environment (after activating your environment) to install additional python
packages:

``` sh
$ pip install SOMEPACKAGE
```

To update a single package with conda:

``` sh
$ conda update -n ENVIRONMENT SOMESOFTWARE
```

or to update all packages:

``` sh
$ conda update -n ENVIRONMENT --all
```

#### In batch/job scripts

To be able to use this environment in a batch script (job script), you will need
to include the following in your batch script, before calling the python program:

```sh
# load the Anaconda3
module load Anaconda3/2019.03

# Set the ${PS1} (needed in the source of the Anaconda environment)
export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}
# So use one of the following lines
# comes with the module load command
# source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate PATH_TO_ENVIRONMENT

# Execute the python program
python pdexample.py
```

### Share your environment

(create_project)=
#### Share with project members on the same machine
By creating conda environments in your project folder
(`conda create -p /cluster/projects/nnXXXXk/conda/ENVIROMENT`)
all your colleagues that are also member of that project have access
 to the environment and can load it with:

``` sh
$ conda activate /cluster/projects/nnXXXXk/conda/ENVIROMENT
```

#### Export your package list

To export a list of all packages/programs installed with conda 
in a certain environment (in this case "ENVIRONMENT"):

``` sh
$ conda list --explicit --name ENVIRONMENT > package-list.txt
```

To setup a new environment (let's call it "newpython")
from an exported package list:

``` sh
$ conda create --name newpython --file package-list.txt
```

Alternatively you can substitute `--name ENVIRONMENT` with `--prefix PATH`.


### Additional Conda information

#### `Disk Quota Exceeded` error message

Conda environments contain a lot of files which can make you exceed your number
of files quota. This happens especially easily when installing conda
environments in your home folder. Check your quota with `dusage`.

To solve this error and reduce your number of files, delete unnecessary and cached files with:

```sh
$ conda clean -a
```

To avoid this error, create your conda environments in your project folder by
using the `--prefix PATH`, see also [here](create_project).

#### Cheatsheet and built-in help

See this [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
for an overview over the most important conda commands.

In case you get confused by the conda commands and command line options
you can get help by adding `--help` to any conda command or have a look
at the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

#### Miniconda vs. Anaconda

Both Miniconda and Anaconda are distributions of the conda repository management
system. But while Miniconda brings just the management system (the `conda`
command), Anaconda comes with a lot of built-in packages.

Both are installed on Saga and Betzy but we advise the use of Miniconda. By
explicitly installing packages into your own environment the chance for
unwanted effects and errors due to wrong or incompatible versions is reduced.
Also you can be sure that everything that happens with your setup is controlled
by yourself.
