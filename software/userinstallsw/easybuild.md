# Installing software with EasyBuild

The Metacenter software team is currently using the
[EasyBuild](https://easybuild.readthedocs.io/en/latest/)
system for installing system-wide software and scientific applications on all
Norwegian HPC systems. It is, actually, quite easy (hence the name) and
straightforward for users to install with the same tool (provided that an
*[easybuild easyconfigs](https://easybuild.readthedocs.io/en/latest/Writing_easyconfig_files.html#what-is-an-easyconfig-file)*
already exists for the particular package you are interested in).

With EasyBuild you might encounter a few different scenarios:

1.  An easyconfig already exists for my software and can be used as is.
2.  An easyconfig exists, but not for the compiler toolchain that I need.
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
reason for the long names, like ``rjags-4-6-intel-2017b-R-3.4.3``
which means that the program ``rjags`` version 4.6 has been built using the
``intel-2017b`` toolchain and aimed at ``R`` version 3.4.3. This in turn means
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

    $ module purge
    $ module avail easybuild
    ----------------------- /cluster/modulefiles/all ------------------------
	EasyBuild/4.3.1

    $ module load EasyBuild/4.3.1

Say you want to install the already mentioned
[rjags](http://cran.r-project.org/web/packages/rjags)
package. Then you first need to find out which easyconfigs are available in the
repository that the EasyBuild module is currently pointing at. This you can
find under the ``$EBROOTEASYBUILD`` path, which becomes active when you load
the EasyBuild module:

    $ ls $EBROOTEASYBUILD/easybuild/easyconfigs/r/rjags
    rjags-4-10-foss-2019b.eb          rjags-4-6-intel-2017a-R-3.4.0.eb  rjags-4-8-foss-2018b-R-3.5.1.eb
    rjags-4-10-foss-2020a-R-4.0.0.eb  rjags-4-6-intel-2017b-R-3.4.3.eb  rjags-4-9-foss-2019a-R-3.6.0.eb

From this list we decide to go for ``rjags`` version mentioned above, i.e.
``rjags-4-6-intel-2017b-R-3.4.3.eb``.

Now, we advice to do an install in three steps, first download the sources of
your software, then do a test run where you check what will be installed and
then the full install.

**Step 1:** To fetch the source, run the following command:

    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch

This will download a tarball into your local EB directory
``.local/easybuild/sources``:

    $ ls $HOME/.local/easybuild/sources/r/rjags
    rjags_4-6.tar.gz

**Step 2:** It may be a good idea to perform a test run to get an overview of
what will be installed with the command you are planning to use. This you get
by the command:

    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --dry-run

This will check that all the necessary dependencies are available in the current
EB repository. If not, you will get an error message. It will also print the
full list of dependencies, where everything marked with ``[x]`` means that the
dependency is already satisfied, and won't be re-installed.

**Step 3:** If the test build was successful, you can perform the build with:

    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --robot --parallel=2

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

    $ ls $HOME/.local/easybuild/software/rjags/4-6-intel-2017b-R-3.4.3/rjags/
    data  DESCRIPTION  help  html  INDEX  libs  Meta  NAMESPACE  R


## Using locally installed software

In the example above we installed the ``rjags`` code locally in our home
folder under ``.local/easybuild/software``. At the same time, EasyBuild
created a new module file under ``.local/easybuild/modules``:

    $ ls .local/easybuild/modules/all/rjags/
    4-6-intel-2017b-R-3.4.3.lua

which can now be loaded and used alongside any other globally installed
software. In order to do so you need to tell the module system to look for
modules in this directory, which is done with the ``module use`` command:

    $ module use $HOME/.local/easybuild/modules/all
    $ module avail rjags
    ------------------------------------ .local/easybuild/modules/all -------------------------------------
        rjags/4-6-intel-2017b-R-3.4.3

    Use "module spider" to find all possible modules.
    Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

You can now load and use the package just like any other module:

    $ module load rjags/4-6-intel-2017b-R-3.4.3

**For more information about the module system, please see:**
<https://lmod.readthedocs.io/en/latest>


## Installing software in a project folder

In order to install semi-globally under a project directory you should make the
following changes to the above procedure.

    $ my-path=/cluster/projects/nnXXXXk/easybuild
    $ mkdir -p $my-path
    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --prefix=$my-path --fetch
    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --prefix=$my-path --dry-run
    $ eb rjags-4-6-intel-2017b-R-3.4.3.eb --prefix=$my-path

where XXXX is your project id number. Note the easybuild folder in the path,
this is a tip for housekeeping and not strictly required. This will give the
path structure as for the local case, with the software and modulefiles
installed in ``cluster/projects/nnXXXXk/easybuild``.

Now the ``rjags`` installation is avaiable to everyone associated with the
``nnXXXXk`` project, after typing:

    $ module use /cluster/projects/nnXXXk/easybuild/modules/all


## Writing your own easyconfigs

Let's say that for some reason you need to use ``rjags`` with the
``intel/2018b`` toolchain, instead of ``2017b`` which we already installed.
A quick look into the EB repo tells you that this particular version is not
available, which means that simply substituting the toolchain version will
*not* work:

    $ eb rjags-4-6-intel-2018b-R-3.4.3.eb --fetch
    == temporary log file in case of crash /tmp/eb-hZoghE/easybuild-2YC1pX.log
    ERROR: Can't find path /cluster/home/$USER/rjags-4-6-intel-2018b-R-3.4.3.eb

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
will copy the ``rjags-4-6-intel-2017b-R-3.4.3.eb`` file and work from there.

**Step 1:** Copy similar easyconfig to somewhere in your ``$HOME``, here
``eb-sandbox``:

    $ cd eb-sandbox
    $ cp $EBROOTEASYBUILD/easybuild/easyconfigs/r/rjags/rjags-4-6-intel-2017b-R-3.4.3.eb .

**Step 2:** Inspect the easyconfig file and check for dependencies; in this
case there are two, ``R`` and ``JAGS``. Next you need to check if any of the
dependencies are available with the toolchain that you want. Here we see that
``R-3.5.1`` *is* available with ``intel-2018b``:

    $ ls $EBROOTEASYBUILD/easybuild/easyconfigs/r/R/*intel-2018b*
    /cluster/software/EasyBuild/4.3.1/easybuild/easyconfigs/r/R/R-3.5.1-intel-2018b.eb

but ``JAGS`` is not:

    $ ls $EBROOTEASYBUILD/easybuild/easyconfigs/j/JAGS/*intel-2018b*
    No such file or directory

However, we do have a version with ``intel/2017b`` (which of course is the one
used by our original ``rjags``), so we'll copy that one as well and adapt it to
our target toolchain:

    $ cp $EBROOTEASYBUILD/easybuild/easyconfigs/j/JAGS/JAGS-4.3.0-intel-2017b.eb .

You can see that this procedure gets exponentially more complicated when you
have to recursively update all dependencies of your original package, but
thankfully in our case it stops here.

```{note}
The versions do not necessarily have to match *literally*, they just need to
not be *conflicting*. For instance, ``intel/2018b`` includes the compiler
``GCCcore/7.3.0``, so any easyconfig with the ``GCCcore-7.3.0`` suffix would
also be compatible.
```

**Step 3:** Starting with your deepest dependency, edit your new easyconfigs
and change the version specifications. In ``JAGS`` it's just a matter of
changing the toolchain to:

    toolchain = {'name': 'intel', 'version': '2018b'}

and at the same time remove the checksum line. For ``rjags`` you do exactly the
same, but you should also update the versions of the dependencies. The ``JAGS``
entry does not have to be changed, because we're still using version 4.3.0, but
``R`` has to be changed to version 3.5.1, which was the one we found in **Step
2** to be available for our toolchain. Note that the version suffix ``-X11...``
is no longer applicable for this version of ``R``, so it should be removed.

**Step 4:** Rename the easyconfig files to match the new versions:

    $ mv JAGS-4.3.0-intel-2017b.eb JAGS-4.3.0-intel-2018b.eb
    $ mv rjags-4-6-intel-2017b-R-3.4.3.eb rjags-4-6-intel-2018b-R-3.5.1.eb

Our new easyconfigs should now look like this:

**JAGS-4.3.0-intel-2018b.eb:**
```{eval-rst}
.. literalinclude:: files/JAGS-4.3.0-intel-2018b.eb
  :language: bash
```

**rjags-4-6-intel-2018b-R-3.5.1.eb:**
```{eval-rst}
.. literalinclude:: files/rjags-4-6-intel-2018b-R-3.5.1.eb
  :language: bash
```

**Step 5:** Build your new module while adding the current directory to
``--robot`` so that your new `.eb` files are picked up:

    $ eb rjags-4-6-intel-2018b-R-3.5.1.eb --robot=. --parallel=2

Again, please don't use too many ``--parallel`` threads on login!

**Step 6:** Load your shining new module:

    $ module use $HOME/.local/easybuild/modules/all
    $ module avail rjags
    --------------------------- $HOME/.local/easybuild/modules/all ---------------------------
        rjags/4-6-intel-2017b-R-3.4.3    rjags/4-6-intel-2018b-R-3.5.1

    $ module load rjags/4-6-intel-2018b-R-3.5.1

**For more information on how to write easyconfigs:**
<https://easybuild.readthedocs.io/en/latest/Writing_easyconfig_files.html>
