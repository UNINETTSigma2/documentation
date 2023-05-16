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
- [Mamba](https://mamba.readthedocs.io/): Fast re-implementation of Coda for
  fast dependency resolution.
- [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
  Ultra-lightweight Mamba which supports the most important `conda` commands.

We typically provide modules for **Anaconda** and **Miniconda**. Either of the
two is fine.


## Typical pitfalls to avoid

```{warning}
- Never use `conda init`.
- Do not modify your `.bashrc` with any Conda stuff or module loads.
- Do not install packages/environments into your home directory otherwise you
  fill your disk quota.
- Do not lose track of what packages you installed. Use `environment.yml` files.
```


### Never use "conda init"



Never use `conda init` on the cluster. This is because it otherwise modifies your `.bashrc`
file and adds the following:
```bash
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

These lines are the reason you see a `(base)` next to your prompt once you log
into the cluster.  If you see `(base)` next to your prompt after a login, you
have a problem.  We recommend to remove those lines from your `.bashrc`.

If you incorrectly activate an environment, Conda itself will complain and suggest "conda init"
but **do not run "conda init"**:
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.
```

Why is `conda init` and modifying `.bashrc` a problem? See right below ...



### Do not modify your .bashrc with any Conda stuff or module loads

The reason for this is that this makes your computations less reproducible for others
and your future self:
- The staff answering your next support request does not have this in their
  `.bashrc`. They will have a different environment and will have a hard time reproducing your problem. The run
  script you send them might produce something different for them.
- Your future self will have a different `.bashrc` (maybe on a different cluster) and will
  have a hard time re-running that calculation.
- Sooner or later your calculations might not work anymore if modules change on the cluster
  during a major upgrade.


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

Normally the only two Conda files that need to be in your home are `~/.conda/environments.txt`
and possibly `~/.condarc`. Everything else should be somewhere else: project folder
or work area.


### Losing track of what packages you installed

If you install packages from the command line using `conda install`, then you
probably won't remember what you installed precisely one year later and this is
a problem for reproducibility for others and your future self trying to publish
that article. Always install from an `environment.yml` file, then you don't
need to remember but it is automatically documented.


## Installing packages/environment from an environment.yml file

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

Once you have such a file you have achieved two things:
- You have documented your dependencies (great for reproducibility)
- You can install from it

To install from that file, you only need to decide where to. **Not into your
home directory**, otherwise you will fill your storage quota.

Here I create a Conda environment from an `environment.yml` file with a script but you
can also do that command after command:
```{code-block} bash
---
emphasize-lines: 8
linenos:
---
#!/usr/bin/env bash

module load Anaconda3/2022.05

#                                 change this
#                                     |
#                                     v
my_conda_storage=/cluster/projects/nn____k/conda

export CONDA_PKGS_DIRS=$my_conda_storage/package-cache
conda env create --prefix $my_conda_storage/myproject --file environment.yml
```

You need to adapt the location (line 8) and also change the name ("myproject").
On line 10 we define `CONDA_PKGS_DIRS` to also be in your well-defined `my_conda_storage`,
otherwise package cache is in your home directory and that is a problem.

If I run this for my example `environment.yml` above it creates the cache and environment folders
and each contains many files already:
```console
$ find . -maxdepth 1 -type d -exec sh -c 'echo -n "{}: "; find "{}" -type f | wc -l' \; | sort -n -k2 -r

.: 31510
./package-cache: 16549
./myproject: 14961
```

35 thousand files! You can easily get up to 100 thousand or more files and that
is too much for our home directories.


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

This is equivalent to installing them one by one into an active environment:
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


## If I change the environment.yml file, do I need to remove everything and restart from scratch?

No need to remove everything. You can adjust your `environment.yml` and all you need to change
is `conda env create` to `conda env update`:
```{code-block} bash
---
emphasize-lines: 11
linenos:
---
#!/usr/bin/env bash

module load Anaconda3/2022.05

#                                 change this
#                                     |
#                                     v
my_conda_storage=/cluster/projects/nn____k/conda

export CONDA_PKGS_DIRS=$my_conda_storage/package-cache
conda env update --prefix $my_conda_storage/myproject --file environment.yml
```


## Should I have one or many environment.yml files?

We recommend one **environment per project**, not one for all projects.  Here
meaning research project/ code project, not compute allocation project.  It is
OK to share an environment with colleagues if they use the same code but it is
a good idea to not try to have an environment for "everything" and all your
many projects.

The reason is that one day you will want to share the environment with somebody
else and the somebody else does not want to install everything to run that one
tiny script that only needs a tiny environment.

Another reason to have one environment per project is that projects can have
different and conflicting dependencies.


## Activating the environment interactively

We need three commands:
```bash
$ module load Anaconda3/2022.05
$ source $EBROOTANACONDA3/bin/activate

$ conda activate /cluster/projects/nn____k/conda/myproject
```

If you used Miniconda instead of Anaconda, then the first two lines change:
```bash
$ module load Miniconda3/22.11.1-1
$ source $EBROOTMINICONDA3/bin/activate

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
[user@login-1.FRAM ~/example]$ module load Anaconda3/2022.05

[user@login-1.FRAM ~/example]$ source $EBROOTANACONDA3/bin/activate

(base) [user@login-1.FRAM ~/example]$ conda activate /cluster/projects/nn____k/conda/myproject

(/cluster/projects/nn____k/conda/myproject) [user@login-1.FRAM ~/example]$ python example.py
```


## Activating the environment in your job script

We activate the environment in the job script the same way we activate it
interactively on the command line (above), only with some additional `SBATCH`
directives on top:
```{code-block} bash
---
emphasize-lines: 13-14, 19
linenos:
---
#!/usr/bin/env bash

#                change this
#                    |
#                    v
#SBATCH --account=nn____k
#SBATCH --job-name=example
#SBATCH --qos=devel
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

# the actual module version might be different
module load Anaconda3/2022.05
source $EBROOTANACONDA3/bin/activate

#                               change this
#                                   |
#                                   v
conda activate /cluster/projects/nn____k/conda/myproject

python --version
python example.py
```

We need three lines before running any code that depends on the packages in
your environment: loading the module, sourcing the activate script, and `conda
activate` our environment.

If you used Miniconda instead of Anaconda, then lines 13 and 14 (above) might
look like this instead (version might be different):
```bash
module load Miniconda3/22.11.1-1
source $EBROOTMINICONDA3/bin/activate
```


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
$ module load Anaconda3/2022.05
$ source $EBROOTANACONDA3/bin/activate
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
