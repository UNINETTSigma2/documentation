# How can I, as a user, install software for myself or my project with EasyBuild?

Currently, the software-team in the Metacenter is using the [EasyBuild system](https://easybuild.readthedocs.io/en/latest/) for installing system-wide software and scientific applications. It is, actually, quite easy (hence the name) and straight forward for users to do install with the same tool.

There are two distinct scenarios covered in this tutorial; first installations for single user only - placing the software in standard user home folder. Second, installations for a project/group - placing the software in a project folder for many to share.

Note that the software install team at the Norwegian Metacenter **never** do installations in `$HOME` for users, and preferably not in /cluster/projects.

## Installing software in home-folder:

Log in to Fram your preferred way. Note that there is, currently at least, no default modules on Fram. Thus you need to be explicit. The easybuild version changes at least a couple of times a year, so do check what is the available version(s) by typing:

	module avail easybuild

Now, as of November 2020, you will see this:

	module avail EasyBuild

	----------------------- /cluster/modulefiles/all ------------------------
	   EasyBuild/4.3.1

	Use "module spider" to find all possible modules.
	Use "module keyword key1 key2 ..." to search for all possible modules
	matching any of the "keys".

Choose 4.3.1 in this case and load this module by typing:

	module load EasyBuild/4.3.1

Now, we advice to do an install in three steps, first download the sources of your software, then do a test run where you check what will be installed and then the full install.

Say you want to install [rjags 4.6](http://cran.r-project.org/web/packages/rjags), you would need to find out which *[easybuild easyconfigs](https://easybuild.readthedocs.io/en/latest/Writing_easyconfig_files.html#what-is-an-easyconfig-file)* that is currently available for that explicit code. This you would find here: <https://github.com/easybuilders/easybuild-easyconfigs/tree/master/easybuild/easyconfigs/r/rjags>. A general overview of all community available *easyconfigs* is available here:<https://github.com/easybuilders/easybuild-easyconfigs/tree/master/easybuild/easyconfigs>. In this example we choose the *easyconfig* for rjags version 4-6 with, compiled with the full intel suite release from second half of 2017 and aimed at R release 3.4.3 (thus the long name of the eb-file).

To download the source of software, type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch

It may be a good idea to get an overview of what will be installed with the command you are planning to use, this you get by the command:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --dry-run

if this proves successful, then type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --robot

Then the process should go absolutely fine, and you will receive a nice little message on the command line stating that installation succeeded.

Now the software and the module(s) you installed are in a folder called ".local". You can inspect it by typing (note the path!)

	cd .local/easybuild

There you should see the following:

	build  ebfiles_repo  modules  software  sources

## Installing software in project folder:

Do as described above regarding login, loading of the EasyBuild module and considerations regarding what to install.

Then do as follows:

	mkdir -p /cluster/projects/nnXXXXk/easybuild
	eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch --prefix=/cluster/projects/nnXXXXk/easybuild
	eb rjags-4-6-intel-2017b-R-3.4.3.eb --dry-run --prefix=/cluster/projects/nnXXXXk/easybuild #as mentioned above.

where XXXX is your project id number. When a successful download of sources is made, then type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --prefix=/cluster/projects/nnXXXXk/easybuild

Note the easybuild folder in the path, this is a tip for housekeeping and not strictly required. This will give the path structure as for the local case, with the software and modulefiles installed in **cluster/projects/nnXXXXk/easybuild**.

A bit more elegant, since the long and horrid path is used thrice, you may do this:

	projsw=/cluster/projects/nnXXXXk/easybuild
	mkdir -p $projsw
	eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch --prefix=$projsw
	eb rjags-4-6-intel-2017b-R-3.4.3.eb --dry-run --prefix=$projsw

###For more advanced settings:

Please check upon options with

	eb --help

or read up on the [EasyBuild documentation](https://easybuild.readthedocs.io/en/latest/) on web.

## Using software installed in non-standard path:

The default path for modulefiles only contains the centrally installed modules. Thus, if you want the modulefilesystem to find the software you installed either for your own usage or on behalf of the project group, you need to make the module-system aware of alternative paths.

**For the case of install in user-home: (still using the rjags example)**

	module use .local/easybuild/modules/all
	module avail rjags #Just to check if it is found
	module load rjags/4-6-intel-2017b-R-3.4.3


**For the case of installing i group/project folder:**

	module use /cluster/projects/nnXXXXk/easybuild/modules/all
	module avail rjags #Just to check if it is found
	module load rjags/4-6-intel-2017b-R-3.4.3

**For more information about the module system, please see:** <https://lmod.readthedocs.io/en/latest/>


# How can I as user install Python packages?

Users can install Python packages in a virtual Python environment. Here is how
you create a virtual environment with the command `virtualenv`:

``` sh
# Create the virtual environment.
virtualenv my_new_pythonenv
# Activate the environment.
source my_new_pythonenv/bin/activate
# Install packages with pip. Here we install pandas.
pip install pandas
```

```{note}
When running software from your Python environment in a batch script, it is
_highly_ recommended to activate the environment _only_ in the script (see
below), while keeping the login environment clean when submitting the job,
otherwise the environments can interfere with each other (even if they are the
same).
```

## Using the virtual environment in a batch script

In a batch script you will activate the virtual environment in the same way as
above. You must just load the python module first:

```
# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

# Load modules
module load Python/3.7.2-GCCcore-8.2.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source my_new_pythonenv/bin/activate

# execute example script
python pdexample.py
```

## Using Anaconda and Miniconda in batch

An alternative to creating virtual environments with virtualenv, is to create
environment with Anaconda or Miniconda. To able to use the conda
environment in a batch script, you will need to source a conda environment
file. First create a environment, here also installing:

```
# load Anaconda3
module load Anaconda3/2019.03

# install pandas as part of creating the environment
conda create -t testenv pandas

# activate the environment
conda activate testenv
```

To be able to use this environment in a batch script, you will need to include
the following in your batch script, before calling the python program:

```
# load the Anaconda3
module load Anaconda3/2019.03

# Set the ${PS1} (needed in the source of the Anaconda environment)
export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} comes with the
# module load command
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate /cluster/home/rnl/anaconda3/envs/testenv

# Execute the python program
python pdexample.py
```
