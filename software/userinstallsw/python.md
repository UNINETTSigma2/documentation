# Installing Python packages

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
file. First create a environment, here also installing pandas:

```
# load Anaconda3
module load Anaconda3/2019.03

# install pandas as part of creating the environment
conda create -t testenv pandas

# activate the environment
conda activate testenv
```

If you are planning on adding many libraries to your environment, you should
consider placing it in a directory other than your $HOME, due to the
{ref}`storage restrictions <clusters-homedirectory>` on that folder. One
alternative could be to use the {ref}`Project area <project-area>`, please
check out {ref}`Storage areas on HPC clusters <clusters-overview>` for other
alternatives. To install conda in an alternative location, add
`-p /path/to/new/directory` to the create environment command, e.g.:

```
conda create -p /path/to/new/directory
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
