(easybuild)=

# Installing software with EasyBuild

The NRIS software team is currently using the
[EasyBuild](https://easybuild.readthedocs.io/en/latest/)
system for installing system-wide software and scientific applications on all
Norwegian HPC systems. It is, actually, quite easy (hence the name) and
straightforward for users to install with the same tool (provided that an
[easyconfig file](https://docs.easybuild.io/writing-easyconfig-files/)
already exists for the particular package you are interested in).

With EasyBuild you might encounter a few different scenarios:

1.  An easyconfig already exists for my software and can be used as is.
2.  An easyconfig exists, but not for the software version or compiler toolchain that I need.
3.  An easyconfig does not exist for my software (not covered here, please
    refer to general EasyBuild tutorials).

In the following we will cover the first two scenarios. We will describe
installations for single users - placing the software in standard user
home folder - as well as installations for an entire project/group - placing
the software in a project folder for many to share.

## About EasyBuild

EasyBuild is a software build and installation framework specifically targeting
HPC systems, with focus on build *automation*, *reproducibility*, and
automatic *dependency* resolution. It is fully compatible with the
[Lmod](https://lmod.readthedocs.io/en/latest/)
module system that is used on all our HPC clusters, and every EB installation
will automatically generate a corresponding module file which allows you to
load the software into your environment.

EasyBuild is very explicit in the build specifications for each part in a
software's chain of dependencies, and is very careful not to mix dependencies
that are built using different compiler versions or toolchains. This is the
reason for the long names, like ``rjags-4-12-foss-2021b-R-4.1.2``
which means that the program ``rjags`` version 4.12 has been built using the
``foss-2021b`` toolchain and aimed at ``R`` version 4.1.2. This in turn means
that all other dependencies that ``rjags`` might have, or anything that want to
*use* ``rjags`` as a dependency, will have to be compatible with these versions.

## Installing software in your home folder

By default, EasyBuild will download, build and install software in a directory
called ``.local/easybuild/`` in your own ``$HOME`` folder, which means that
only you will be able to find and use the package after it is installed. The
following procedure applies equally well to all our current HPC systems.

The first thing we do is to load the EasyBuild module on a login node of your
cluster of choice. Make sure that you have a clean environment with no other
modules loaded, then search for available EasyBuild versions and load the
latest version you can find:

    $ module reset
    $ module avail easybuild
    ----------------------- /cluster/modulefiles/all ------------------------
	EasyBuild/4.9.0

    $ module load EasyBuild/4.9.0

Say you want to install the already mentioned
[rjags](https://cran.r-project.org/web/packages/rjags) package. Then you first
need to find out which easyconfigs are available on the system.  This can be
done with the `--search-filename` or `-S` option. So to display all rags
version that can be installed use:

    $ eb --search-filename rjags
    == found valid index for /cluster/software/EasyBuild/4.9.0/easybuild/easyconfigs, so using it...
     * rjags-4-6-intel-2017a-R-3.4.0.eb
     * rjags-4-6-intel-2017b-R-3.4.3.eb
     * rjags-4-8-foss-2018b-R-3.5.1.eb
     * rjags-4-9-foss-2019a-R-3.6.0.eb
     * rjags-4-10-foss-2019b.eb
     * rjags-4-10-foss-2020a-R4.0.0.eb
     * rjags-4-10-foss-2020b-R-4.0.3.eb
     * rjags-4-10-foss-2020b-R-4.0.4.eb
     * rjags-4-10-foss-2020b-R-4.0.5.eb
     * rjags-4-10-foss-2021a-R-4.1.0.eb
     * rjags-4-10-fosscuda-2020b-R-4.0.3.eb
     * rjags-4-10-fosscuda-2020b-R-4.0.4.eb
     * rjags-4-10-fosscuda-2020b-R-4.0.5.eb
     * rjags-4-12-foss-2021b-R-4.1.2.eb
     * rjags-4-13-foss-2022a-R-4.2.1.eb
     * rjags-4-13-foss-2022b-R-4.2.2.eb

    Note: 1 matching archived easyconfig(s) found, use --consider-archived-easyconfigs to see them


From this list we decide to go for ``rjags`` version mentioned above, i.e.
``rjags-4-12-foss-2021b-R-4.1.2.eb``.

Now, we advice to do an install in three steps, first download the sources of
your software, then do a test run where you check what will be installed and
then the full install.

**Step 1:** To fetch the source, run the following command:

    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --fetch

This will download a tarball into your local EB directory
``.local/easybuild/sources``:

    $ ls $HOME/.local/easybuild/sources/r/rjags
    rjags_4-12.tar.gz

**Step 2:** It may be a good idea to perform a test run to get an overview of
what will be installed with the command you are planning to use. This you get
by the command:

    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --dry-run

This will check that all the necessary dependencies are available in the current
EB repository. If not, you will get an error message. It will also print the
full list of dependencies, where everything marked with ``[x]`` means that the
dependency is already satisfied, and won't be re-installed. If you only want to see
which extra dependencies are going to be installed:

    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --missing-modules

**Step 3:** If the test build was successful, you can perform the build with:

    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --robot --parallel=2

where ``--robot`` means that EasyBuild should automatically resolve and install
all necessary dependencies, and ``--parallel`` will set the number of CPU
threads to use in the build. Hopefully, this returns successfully after a few
minutes with a message like this:

    == Build succeeded for 1 out of 1
    == Temporary log file(s) /tmp/eb-BoOCuj/easybuild-CuSy5M.log* have been removed.
    == Temporary directory /tmp/eb-BoOCuj has been removed.

```{note}
The default build will use *all* available cores, so *please* set the
``--parallel`` option to a more reasonable number to avoid clogging the
login node.
```

You can now confirm that the package has been installed under
``.local/easybuild/software``:

    $ ls $HOME/.local/easybuild/software/rjags/4-12-foss-2021b-R-4.1.2/rjags/
    data  DESCRIPTION  help  html  INDEX  libs  Meta  NAMESPACE  R


## Using locally installed software

In the example above we installed the ``rjags`` code locally in our home
folder under ``.local/easybuild/software``. At the same time, EasyBuild
created a new module file under ``.local/easybuild/modules``:

    $ ls .local/easybuild/modules/all/rjags/
    4-12-foss-2021b-R-4.1.2.lua

which can now be loaded and used alongside any other globally installed
software. In order to do so you need to tell the module system to look for
modules in this directory, which is done with the ``module use`` command:

    $ module use $HOME/.local/easybuild/modules/all
    $ module avail rjags
    ------------------------------------ .local/easybuild/modules/all -------------------------------------
        rjags/4-12-foss-2021b-R-4.1.2

    Use "module spider" to find all possible modules.
    Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

You can now load and use the package just like any other module:

    $ module load rjags/4-12-foss-2021b-R-4.1.2

**For more information about the module system, please see:**
<https://lmod.readthedocs.io/en/latest>

If you are planning to install more than a few modules you will quickly run out of disk quota in the home folder. In general we recommend that you instead install in a project folder.

## Installing software in a project folder

In order to install semi-globally under a project directory you should make the
following changes to the above procedure.

    $ my_path=/cluster/projects/nnXXXXk/easybuild
    $ mkdir -p $my_path
    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --prefix=$my_path --fetch
    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --prefix=$my_path --dry-run
    $ eb rjags-4-12-foss-2021b-R-4.1.2.eb --prefix=$my_path

where XXXX is your project id number. Note the easybuild folder in the path,
this is a tip for housekeeping and not strictly required. This will give the
path structure as for the local case, with the software and modulefiles
installed in ``cluster/projects/nnXXXXk/easybuild``.

Now the ``rjags`` installation is available to everyone associated with the
``nnXXXXk`` project, after typing:

    $ module use /cluster/projects/nnXXXk/easybuild/modules/all

The prefix option (in fact any command line option) can also be set vi an
environment variable like this:

    $ export EASYBUILD_PREFIX=/cluster/project/nnXXXXk/easybuild

and then you need not repeat the ``--prefix`` option on the command line.

## Writing your own easyconfigs

Let's say that for some reason you need to use ``rjags-4-12`` with the
``foss/2022a`` toolchain, instead of ``2021b`` which we already installed.
A quick look into the EB repo tells you that this particular version is not
available, which means that simply substituting the toolchain version will
*not* work:

    $ eb rjags-4-12-foss-2022a-R-4.1.2.eb --fetch
    == Temporary log file in case of crash /tmp/eb-bliwuikc/easybuild-2a9dc17d.log
    == found valid index for /cluster/software/EasyBuild/4.9.0/easybuild/easyconfigs, so using it...
    ERROR: One or more files not found: rjags-4-12-foss-2022a-R-4.1.2.eb (search paths:
    /cluster/software/EasyBuild/4.9.0/easybuild/easyconfigs)

In this case what you need to do is to write your own easyconfig file and use
that for your easy build. Depending on the package this can be either really
simple or frustratingly complicated. This particular example is somewhere in
between.

Now, be aware that our clone of the EB repo will only occasionally be updated,
so you might get lucky and find that the exact version you are looking for has
already become available on the
[central EB repo](https://github.com/easybuilders/easybuild-easyconfigs/tree/master/easybuild/easyconfigs),
so please check this out first. In that case you can simply download or copy
the easyconfig file from there, otherwise it's a good idea to start from an
easyconfig that is similar to the one you are trying to make. In our case we
will copy the ``rjags-4-12-foss-2021b-R-4.1.2.eb`` file and work from there.

**Step 1:** Copy similar easyconfig to somewhere in your ``$HOME``, here
``eb-sandbox``:

    $ cd eb-sandbox
    $ eb --copy-ec rjags-4-12-foss-2021b-R-4.1.2.eb

**Step 2:** Inspect the easyconfig file and check for dependencies; in this
case there are two, ``R`` and ``JAGS``. Next you need to check if any of the
dependencies are available with the toolchain that you want. Here we see that
``R-4.2.1`` *is* available with ``foss-2022a``:

    $ ls $EBROOTEASYBUILD/easybuild/easyconfigs/r/R/*foss-2022a*
    /cluster/software/EasyBuild/4.9.0/easybuild/easyconfigs/r/R/R-4.2.1-foss-2022a.eb

Also a never version of ``JAGS``is available, but if we assume that we need to keep the version at 4.3.0 it is not available for the new toolchain

    $ ls $EBROOTEASYBUILD/easybuild/easyconfigs/j/JAGS/JAGS-4.3.0-foss-2022a*
    No such file or directory

However, we do have a version with ``foss/2021b`` (which of course is the one
used by our original ``rjags``), so we'll copy that one as well and adapt it to
our target toolchain:

    $ eb --copy-ec JAGS-4.3.0-foss-2021b.eb

You can see that this procedure gets exponentially more complicated when you
have to recursively update all dependencies of your original package, but
thankfully in our case it stops here.

```{note}
The toolchains do not necessarily have to match *literally*, they just need to
not be *conflicting*. For instance, ``foss/2022a`` includes the compiler
``GCCcore/11.3.0``, so any easyconfig with the ``GCCcore-11.3.0`` suffix would
also be compatible. For more information about common toolchains: <https://docs.easybuild.io/common-toolchains/>
```

**Step 3:** Starting with your deepest dependency, edit your new easyconfigs
and change the version specifications. In ``JAGS`` it's just a matter of
changing the toolchain to:

    toolchain = {'name': 'foss', 'version': '2022a'}

If the software version is changed you also need to update the checksum line. 
For ``rjags`` you do exactly the
same, but you should also update the versions of the dependencies. The ``JAGS``
entry does not have to be changed, because we're still using version 4.3.0, but
``R`` has to be changed to version 4.2.1, which was the one we found in **Step
2** to be available for our toolchain.

**Step 4:** Rename the easyconfig files to match the new versions:

    $ mv JAGS-4.3.0-foss-2021b.eb JAGS-4.3.0-foss-2022a.eb
    $ mv rjags-4-12-foss-2021b-R-4.1.2.eb rjags-4-12-foss-2022a-R-4.2.1.eb

Our new easyconfigs should now look like this:

**JAGS-4.3.0-foss-2022a.eb:**
```{eval-rst}
.. literalinclude:: files/JAGS-4.3.0-foss-2022a.eb
  :language: bash
```

**rjags-4-12-foss-2022a-R-4.2.1.eb:**
```{eval-rst}
.. literalinclude:: files/rjags-4-12-foss-2022a-R-4.2.1.eb
  :language: bash
```

**Step 5:** Build your new module while adding the current directory to
``--robot`` so that your new `.eb` files are picked up:

    $ eb rjags-4-12-foss-2022a-R-4.2.1.eb --robot=. --parallel=2

Again, please don't use too many ``--parallel`` threads on login!

**Step 6:** Load your shining new module:

    $ module use $HOME/.local/easybuild/modules/all
    $ module avail rjags
    --------------------------- $HOME/.local/easybuild/modules/all ---------------------------
        rjags/4-12-foss-2021b-R-4.1.2    rjags/4-12-foss-2022a-R-4.2.1

    $ module load rjags/4-12-foss-2022a-R-4.2.1

```{note}
In this particular case we could also have used the ``--try-*`` options to update the toolchain 
version without having to edit any easyconfigs by hand: <https://docs.easybuild.io/using-easybuild/#tweaking_easyconfigs_using_try> 
```

**For more information on how to write easyconfigs:**
<https://easybuild.readthedocs.io/en/latest/Writing_easyconfig_files.html>
