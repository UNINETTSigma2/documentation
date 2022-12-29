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

Conda (either in the form of Miniconda or Anaconda) is a combination of a package manager like `pip` and a virtual environment manager like `venv`. You can use it to install and manage many python and non-python packages yourself.

See {ref}`installing-with-conda` for details and examples. 
