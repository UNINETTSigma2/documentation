(installing-python-packages)=

# Installing Python packages as a user

The easiest way to install Python packages as a user is to use `pip` inside a virtual environment.
It is advised to use virtual environments since it is a straight forward way to isolate different
installations from each other. This makes it possible to have multiple versions of the same packages
installed in your `$HOME` without problems of conflicting dependencies.

`pip` is the main package installer for Python and included in every Python installation. It is easy
to use and can be combined with `venv` to manage independent environments. It is recommended that
you at least have one virtual environment for each disparate experiment.

```{note} Key takeaways:

- Always install packages inside a virtual environment

- Do **not** install packages with `pip install --user`. They will end up in `$HOME/.local` and from
  there they will leak into both containers and environments, thus breaking compatibility for other
  installations.

- Leverage the existing software stack for dependencies using the option `--system-site-packages`

- When loading dependencies from the central software stack, always use the same toolchain (more
  info on this further down the text)
```

## Creating a Python virtual environment

In this example we have used `venv` which comes with with the Python standard library. In other
guides/documentation `virtualenv` is used. The first is a subset of the latter and has all the
functionality we need.

First load a Python module with (use `module avail Python` to see all):

```console

$ module load Python/3.8.6-GCCcore-10.2.0
```

Create a virtual environment in your `$HOME` folder with an appropriate name. A good idea here is to
add a suffix like "-env" to the name as it makes it transparent as to what we are dealing with:

```console
$ python3 -m venv $HOME/pandas-env --system-site-packages
```

Activate the environment:

```console
$ source $HOME/pandas-env/bin/activate
```

Install packages with pip. Here we install the Python package pandas:

```console
(pandas-env)$ python3 -m pip install pandas
```

You are now ready to use the new environment. When you are done and want to get out of the
environment, you simply type:

```console
$ deactivate
```

**NB!** Remember that you will always have to load the same module(s) before you activate your environment
next time.

For more information, have a look at the official
[pip](https://pip.pypa.io/en/stable/) and
[venv](https://docs.python.org/3/library/venv.html) documentations.

```{note}
When running software from your Python environment in a batch script, it is _highly_ recommended to
activate the environment _only_ in the script (see below), while keeping the login environment clean
when submitting the job, otherwise the environments can interfere with each other (even if they are
the same).
```

## Choosing a Python version

If you need a specific version of Python for your installation, then you can search and see if that
version is available on our system. This command will give you a list of all Python modules
installed:

```console
$ module avail python
```

Load the module you need and then create a virtual environment before you start installing packages
inside of it.

## Leveraging the existing pool of Python packages

The dependencies that are needed to install a certain Python package are usually listed in the
`requirements.txt` file. This file is found in the sourcefiles for the package you are interested
in. The dependencies can sometimes be found under the variable `install_requires` of the file
`setup.py` (also found in the sourcefiles).

Since we already have hundreds of Python packages installed (in different versions) on our systems,
you can utilize those when installing the package you need using this small procedure:

1. Search to see if any of your dependencies are available using `module spider`
2. Make sure the modules (which contain your dependencies) are built with the same toolchain (more on
 this further down)
3. Load all the modules you need (if you do not get an error message, they are compatible)
4. Create your virtual environment and install the Python package you need

### Searching for dependencies 

If you already know some of the Python packages you want to use, you can search for them directly
with the `module spider` command. Let us say you want to use the Python package `numpy`:

```console
$ module spider numpy

-----------------------------------------------------------------------------------------------------------------------------------------
  numpy:
-----------------------------------------------------------------------------------------------------------------------------------------
     Versions:
        numpy/1.25.1 (E)
        numpy/1.26.2 (E)
        numpy/1.26.4 (E)
```

This will give you a list of all version of `numpy` installed. In order to see what module contains
the version of `numpy` you need, run `module spider` again with the version number:

```console
$ module spider numpy/1.26.4

-----------------------------------------------------------------------------------------------------------------------------------------
  numpy: numpy/1.26.4 (E)
-----------------------------------------------------------------------------------------------------------------------------------------
    This extension is provided by the following modules. To access the extension you must load one of the following modules. Note that any module names in parentheses show the module location in the software hierarchy.

       SciPy-bundle/2024.05-gfbf-2024a
```

In order to see what other Python packages you will get access to when loading this `SciPy-bundle` module, run this command:

```console
$ module spider SciPy-bundle/2024.05-gfbf-2024a
     Included extensions
      ===================
      beniget-0.4.1, Bottleneck-1.3.8, deap-1.4.1, gast-0.5.4, mpmath-1.3.0,
      numexpr-2.10.0, numpy-1.26.4, pandas-2.2.2, ply-3.11, pythran-0.16.1,
      scipy-1.13.1, tzdata-2024.1, versioneer-0.29
```

If you then load the module you can check which version of Python it comes with:

```console
[ec-parosen@login-3 ~]$ module load SciPy-bundle/2024.05-gfbf-2024a
[ec-parosen@login-3 ~]$ which python3
/cluster/software/EL9/easybuild/software/Python/3.12.3-GCCcore-13.3.0/bin/python3
```

We see here that `SciPy-bundle/2024.05-gfbf-2024a` is built on top of the
`Python/3.12.3-GCCcore-13.3.0` module. You only need to load the first since the latter is a
dependecy and will be loaded automatically.

### Choosing a toolchain

All modules on our systems are built using a compiler (C/C++/Fortran) and a few different common
libraries (OpenMPI, OpenBLAS, FFTW etc). Together they form something called a toolchain and the
most common one is the `foss` (Free and Open Source Software) toolchain. As newer version of said
tools are released, a new toolchain version is also released (usually twice a year).


```{note}
If you want to combine several different modules that contain the Python packages you need, they
all need to come from the same **toolchain**. For example, you load modules either from the
`foss/2023a` toolchain or the `foss/2022b` toolchain. Note that `GCCcore-12.3.0` is a subtoolchain
of `foss/2023a` and modules with either one of these postfixes are thus comptatible. Here is a list
of all installed foss toolchains and the GCC versions included in them:
```

```
foss/2021a -> 10.3.0
foss/2021b -> 11.2.0
foss/2022a -> 11.3.0
foss/2022b -> 12.2.0
foss/2023a -> 12.3.0
foss/2023b -> 13.2.0
foss/2024a -> 13.3.0
```

You can read more about toolchains in the official
[EasyBuild](https://docs.easybuild.io/common-toolchains/?#common_toolchains_foss) documentation. 


## Using the virtual environment in a batch script

In a batch script you will activate the virtual environment in the same way as
above. You must just load the python module first:

```
# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

# Load modules
module load Python/Python/3.12.3-GCCcore-13.3.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source $HOME/my_new_pythonenv/bin/activate

# execute example script
python pdexample.py
```

## Sharing your Python package configuration with others

To allow other researchers to replicate your virtual environment setup it can be
a good idea to "freeze" your packages. This tells `pip` that it should not
silently upgrade packages and also gives a good way to share the exact same
packages between researchers.

To freeze the packages into a list to share with others run:
```console
$ python3 -m pip list --format freeze > requirements.txt

```

The file `requirements.txt` will now contain the list of packages installed in
your virtual environment with their exact versions. When publishing your
experiments it can be a good idea to share this file which other can install in
their own virtual environments like so:

```console
$ python -m pip install -r requirements.txt
```

Your virtual environment and the new one installed from the same
`requirements.txt` should now be identical and thus should replicate the
experiment setup as closely as possible.
