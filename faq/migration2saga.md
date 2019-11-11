# Migration to Saga

In general, the user environment on Saga is designed to be as similar as possible
to the one on Fram. Users coming from Abel or Stallo will need to adopt a little to the different
queue system setup and to the newer software module system.

Before you move existing scripts, jobs, etc to Saga, we recommend that you make
yourself familiar with Saga by reading this page, work on the exercises listed and
explore Saga's capabilities by trying out new commands and options.

The major steps in migrating to Saga are:

*  Getting an account and project quota on Saga.
*  Getting aware of differences (disk quota, module system, job types, running jobs, how to get help, no backup currently).
*  Transfering data, scripts etc from other machines to Saga.
*  Modifying scripts & routines to match differences on Saga.
*  **Verifying that your jobs run efficiently and produce the same results as on other systems!**
*  Be patient with user support [(support@metacenter.no)](mailto:support@metacenter.no), but don't hesitate to ask questions!

## Current (major) differences of Saga

*  **NO backup of any file system, neither $HOME, shared folders, project folders - please, backup important data yourself until we've implemented it!**
*  **NIRD storage is not mounted on Saga's login nodes!**
*  **Quota/disk usage policies are not enforced yet! We strongly recommend to not use more than the quota of 20 GiB on $HOME or you may run into problems later on!**

All of this will be changed shortly.

## Getting familiar with Saga

Please, read the [introduction about Saga](../quick/saga.md) to inform yourself about Saga's
characteristics.

## Account and projects

You need a user account (say `USER`) and a project account on Saga before you can start using it. Obtaining
a user account and a project account on Saga works as for the other systems (except that for
the Notur period 2019.2 one could not directly apply for compute time on Saga - instead
you or your PI will be informed when you can start migrating to Saga). For details see
[Get User Account](https://www.metacenter.no/user/application/) and
[Apply for e-infrastructure resources](https://www.sigma2.no/content/apply-e-infrastructure-resources).

## Login

Use your favourite login tool, e.g., ssh or Putty, to login with your user account
into Saga. The machine name for Saga is `saga.sigma2.no`. An example command to login
with ssh is

`ssh USER@saga.sigma2.no`

For more information see [Getting Started](../quick/gettingstarted.md). Note, on
Saga there is no support for remote desktops configured yet.

## File systems and storage quota

**Note! Currently (september 2019), there is no backup of any file system on Saga!**

Users on Saga have access to the following file systems and usage policies:

| Filesystem | Access | Usage policy (quota, removal) |
| ------ | ------ |
| home folder | `$HOME` | 20 GiB quota |
| user specific work folder | `$USERWORK` | any file older than 21/42 days will be removed automatically, 21 days when total disk usage (do `df -h /cluster` to check) is above 69 % | 
| project folder | `/cluster/projects/nnXXXk` | quota depending on grant |
| job scratch | `$SCRATCH` | specific to each running job, will be removed at end of job |

Check your disk usage with `du -sh $HOME`. Replace `$HOME` with any other folder
you want to check.

**Exercise 1:** Check your disk usage of `$HOME` and `$USERWORK`.

## Module system

Saga uses **lmod** as module system - the same as on Fram. One major difference
to Abel's older module system is that it is case-insensitive (see Exercise 3 below).

Use

`module avail`

to list all available modules. If you want to check which versions of a specific
package are available do

`module avail SAM`

which shows all packages whose name/version matches `SAM`.

**Exercise 2:** Play around with `module avail` to identify all packages built with the
Intel compiler.

**Exercise 3:** Try `module avail NETCDF` and `module avail netCDF` on both Saga
and Abel and compare the differences. 

You may notice that the `name/version` identifiers for modules on Saga differ from
other systems, particularly when you're used to the module system on Abel.
On Saga (and Fram) the *version* (the part following the `/`) decodes both the
version of the software package and the tool chain (decodes a compiler version and
supporting libraries) used to build the package.
The structure of these identifiers is the same on Saga and Fram. While the modules
system on Abel is following a similar concept (`name/version`) as identifier, the *tool chain*
part usually contains only a brief keyword for the compiler being used.

## Project accounts and compute hour usage

The commands `projects` and `cost` work in the same manner as on Fram and Abel.
The former lists all *project* accounts (don't confuse with your user account)
you have access to. The latter provides details about allocated, used, running,
pending and available cpu hours for your project accounts. For details of
these commands check their usage information `projects --help` and `cost --help`.

**Exercise 4:** Determine all projects you have access to on Saga and Abel/Fram.

**Exercise 5:** Check how many cpu hours are available in the project accounts
you have access to.

## Interactive jobs

Interactive jobs provide you a shell prompt on one of the compute nodes where
you can run any commands interactively. This is useful for several scenarios
such as debugging issues, exploring/developing complex jobs composed of several
steps, determining memory requirements, initial trials to build software packages,
etc. Common to these scenarios is the need for more than one core or that they are
compute-intensives and/or run for longer times (longer than a few minutes) - in
other words activities which are forbidden on the login nodes (we monitor to detect
usage exceeding low thresholds and automatically kill processes which violate the
usage policy).

Once you have found working sequences of commands, for example, for complex jobs
or for building software packages it is highly recommended to put such sequences
into a shell script and execute that as a batch job. An example command to
submit an interactive job is

`srun --account=nnXXXXk --time=00:01:00 --mem-per-cpu=1G --pty bash -i`

The parameters before `--pty bash -i` could also be used to submit a batch job
(often they are put into a shell script). On Abel you may have used the command
`qlogin` to submit an interactive job.

Our documentation provides additional information about [interactive jobs](../jobs/interactive_jobs.md).
You may also check the manual page for srun with the command `man srun`.

**Exercise 6:** Determine the compute node on which the job runs and list all modules
which are loaded once the job has started.

**Exercise 7:** Play with the `srun` command changing the wall time limit (`--time`)
and specifying that it should run on three nodes (parameter `--nodes`).

**Exercise 8:** What happens to your interactive job if your connection to Saga
terminates? Does the job continue to run (as a batch job would do) or is it
terminating (as a shell would do)?

## Batch jobs

Most production jobs are run as batch jobs. You specify how many resources the job
needs, for how long, on which project account it should be billed for, and what
commands the job should run. After you submitted a job, you don't need to do anything
to launch it - it may start at any time when the resources it requests become available.
When this sounds familiar to you being a user on Abel or Fram, then you're right.

However, there are a few changes you have to apply to your existing job scripts.

### Porting job scripts from Abel

A typical job script on Abel looks like (taken from
[A Simple Serial Job](https://www.uio.no/english/services/it/research/hpc/abel/help/user-guide/job-scripts.html#A_Simple_Serial_Job))

    #!/bin/bash

    # Job name:
    #SBATCH --job-name=YourJobname
    #
    # Project:
    #SBATCH --account=YourProject
    #
    # Wall clock limit:
    #SBATCH --time=hh:mm:ss
    #
    # Max memory usage:
    #SBATCH --mem-per-cpu=Size

    ## Recommended safety settings:
    set -o errexit # Make bash exit on any error
    set -o nounset # Treat unset variables as errors

    ## Set up job environment:
    source /cluster/bin/jobsetup
    module purge   # clear any inherited modules
    module load SoftWare/Version

    # It is also recommended to to list loaded modules, for easier debugging:
    module list

    ## Copy input files to the work directory:
    cp MyInputFile $SCRATCH

    ## Make sure the results are copied back to the submit directory:
    chkfile MyResultFile

    ## Do some work:
    cd $SCRATCH
    YourCommands

This job ported to Saga would be (with some additions taken from [Sample MPI Batch Script](../jobs/saga_sample_mpi_job.md))

    #!/bin/bash

    # Job name:
    #SBATCH --job-name=YourJobname
    #
    # Project:
    #SBATCH --account=YourProject
    #
    # Wall clock limit:
    #SBATCH --time=hh:mm:ss
    #
    # Max memory usage: Size is a number plus M (megabyte) or G (gigabyte), e.g., 3M or 5G
    #SBATCH --mem-per-cpu=Size
    #
    # Number of tasks (cores): this is added to make it easier for you to do exercises
    #SBATCH --ntasks=1

    ## Recommended safety settings:
    set -o errexit # Make bash exit on any error
    set -o nounset # Treat unset variables as errors

    ## Set up job environment: (this is done automatically behind the scenes)
    ## (make sure to comment '#' or remove the following line 'source ...')
    # source /cluster/bin/jobsetup
    
    module purge   # clear any inherited modules
    module load SoftWare/Version #nb: 'Version' is mandatory! There are no default versions of modules on Saga!

    # It is also recommended to to list loaded modules, for easier debugging:
    module list

    ## Copy input files to the work directory:
    cp MyInputFile $SCRATCH

    ## Make sure the results are copied back to the submit directory:
    # chkfile MyResultFile
    # chkfile is replaced by 'savefile' on Saga
    savefile MyResultFile

    ## Do some work:
    cd $SCRATCH
    YourCommands

**Exercise 9:** Explore job limits (what amount of resources you can request and
for how long) by adapting the above job script's walltime limit (`--time`), amount
of memory (`--mem-per-cpu`) and number of cores (`--ntasks`). You can change the values
in a script and submit it with just `sbatch SCRIPTNAME` or if you don't want to edit
the script for all trials, add the parameters to the sbatch command, for example,

`sbatch --time=00:01:00 --ntasks=2 --mem-per-cpu=2G SCRIPTNAME`

**Exercise 10:** Add commands to your job script which print environment variables (`env`),
the current date (`date`), sleeps for a while (`sleep 300`) or calculates disk usage
of various directories your job has access to (`du -sh $HOME $USERWORK $SCRATCH`).

**Exercise 11:** Add parameters to get notified when your job is started, ends,
etc. (`#SBATCH --mail-type=ALL` and `#SBATCH --mail-user=YOUR_EMAIL_ADDRESS`).

## Transferring files to Saga

The recommended way to transfer files is `rsync`. In case a transfer is
interrupted or when you have changed files at the origin, you can synchronise files
on Saga by simply rerunning `rsync`. A typical rsync command line looks as follows

`rsync -a -v all_my_scripts_for_paper_x YOUR_USERNAME@saga.sigma2.no:from_abel/.`

Parameter `-a` instructs rsync to copy the whole directory tree starting with
`all_my_scripts_for_paper_x`. Parameter `-v` instructs rsync to be verbose, i.e.,
it will print what it is doing.

**We recommend that you use a specific folder on Saga for files originating from
other systems. That makes it easier to keep an original version and lowers the risk
of overwriting other things you have on Saga.**

**Exercise 12:** On Abel or your own Linux-based machine create a sample directory
tree (for example, by running
`mkdir -p rsync10/A; mkdir -p rsync10/B; touch rsync10/foo rsync10/A/bar rsync10/B/foobar`)
and rsync this to your `$HOME` on Saga.

**Exercise 13:** Rsync this to another directory you have access to, e.g.,
`$USERWORK` or a project directory.

**Exercise 14:** Rsync a larger directory tree to Saga, interrupt it (press `CTRL+C`)
and rerun the rsync command.

## Installing software
Installing software in folders accessible to users, e.g., $HOME, can be relatively
easy with EasyBuild which is also used for system-wide installations. We illustrate
how you can do this for the software package SPAdes for which Saga does not provide
all the versions you would find on Abel (3.1.1, 3.5.0, 3.6.0, 3.7.0, 3.8.0, 3.9.0, 3.10.0, 3.10.1, 3.11.0, 3.11.1, 3.12.0, 3.13.0).

We are going to demonstrate how to install version 3.12.0 which is the last version before the current default on Abel.

On a login node, run `screen -S spades_eb` - see Exercise 16 below for working
with screens. With screens you can detach from and reattach to a running
session, which is particularly useful when your network connection to Saga could
be lost. If that happens, you just have to login to the machine where you started
`screen` and reattach to a session. Without screens (or similar tools) your session
would terminate, which means that also your interactive job would terminate.

Next, start an interactive job (see details
[above](#interactive-jobs)) with

`srun --account=nnXXXXk --time=08:00:00 --nodes=1 --ntasks-per-node=40 --mem=185G --pty bash -i`

Restore a clean module environment

`module restore system`

Load the module for EasyBuild

`module load EasyBuild/3.9.3`

Download an easyconfig file for SPAdes 3.12.0

`eb SPAdes-3.12.0-foss-2018b.eb --fetch`

You may check which easyconfig files are available at [EasyBuild config files](https://github.com/easybuilders/easybuild-easyconfigs/tree/master/easybuild/easyconfigs) and which have been downloaded to Saga at the local directory
`/cluster/software/EasyBuild/3.9.3/lib/python2.7/site-packages/easybuild_easyconfigs-3.9.3-py2.7.egg/easybuild/easyconfigs/`.

Do a dry run

`eb SPAdes-3.12.0-foss-2018b.eb --dry-run`

Assuming that is successful, i.e., no errors reported, build the software - this
may take very long (hours) particularly when many dependencies are built.

`eb SPAdes-3.12.0-foss-2018b.eb -r`

When you run `module avail SPAdes` it may not be shown yet. That's because the
module system doesn't search your $HOME for modules. Do

`module use $HOME/.local/easybuild/modules/all`

Now you should see it

    $ module avail SPAdes

    ------------- /cluster/home/YOUR_USERNAME/.local/easybuild/modules/all --------------
       SPAdes/3.12.0-foss-2018b

    
    ------------------------------ /cluster/modulefiles/all ------------------------------
       SPAdes/3.13.0-foss-2018b    SAMtools/3.13.1-GCC-8.2.0-2.31.1
    ...

**Exercise 15:** Install the easyconfig `SAGE-6.4.eb`. Load it and find out what it provides.

**Exercise 16:** Detach from the screen (type `CTRL+a` and `d`), list running
screens `screen -list` and reattach to screen `spades_eb` with the command `screen -dR spades_eb`.

## Transferring files back home
Sometimes you may need to transfer files out of Saga, e.g., to your laptop or
another server or cluster. The easiest way to do that is to use again `rsync`. If
you can login to the destination machine, then you can simply do

`rsync -a -v my_folder_on_Saga YOUR_USERNAME_ON_YOUR_MACHINE@YOUR_MACHINE:from_saga`

In case you cannot login into machine from Saga, you can initiate the transfer from
your machine. On your machine do

`rsync -a -v YOUR_USERNAME_ON_SAGA@saga.sigma2.no:my_folder_on_Saga from_saga`
