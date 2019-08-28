# Migration to Saga

This page summarizes major steps you need to do when you migrate to Saga.

## Getting familiar with Saga

Please, read the [introduction about Saga](quick/saga.md) to inform yourself about its
characteristics.

## Account and projects

You need an account and a project on Saga before you can start using it. Obtaining
an account and a project on Saga works as for the other systems (except that for
the Notur period 2019.2 one could not directly apply for quota on Saga).

## Login

Use your favourite login tool, e.g., ssh or Putty, to login with your Notur username
into Saga. The machine name for Saga is `saga.sigma2.no`. An example command to login
with ssh is

`ssh me@saga.sigma2.no`

For more information
see [Getting Started](quick/gettingstarted.md). Note, on Saga there is no support
for remote desktop configured yet.

## Module system

Saga uses lmod as module system. One major difference to Abel's older module
system is that it is case-insensitive. Will come back to this below.

Use

`module avail`

to list all available modules. If you want to check which versions of a specific package
are available do

`module avail SAM`

which shows all packages whose name/version matches `SAM`.

**Exercise 1:** Play around with `module avail` to identify all packages built with the
Intel compiler.

**Exercise 2:** Try `module avail NETCDF` and `module avail netCDF` on both Saga and Abel and compare the differences. 

You may notice that the `name/version` identifiers for modules on Saga differ from
other systems, particularly when you're used to the module system on Abel.
On Saga (and Fram) the *version*
(part following the `/`) decodes both the version of the software package and the
tool chain (decodes a compiler version and supporting libraries) used to build the package.
The structure of these identifiers is the same on Saga and Fram. While the modules
system on Abel is following a similar concept (`name/version`) as identifier, the *tool chain*
part usually contains only a brief keyword for the compiler being used. As an example,

## Project accounts and compute hour usage
The commands `projects` and `cost` work in the same manner as on Fram and Abel.
The former lists all *project* accounts (don't confuse with your user account)
you have access to. The latter provides details about allocated, used, running,
pending and available cpu hours for your project accounts. For details of
these commands check their usage information `projects --help` and `cost --help`.

**Exercise 3:** Determine all projects you have access to on Saga and Abel/Fram.

**Exercise 4:** Check how many cpu hours are available in the project accounts
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

`srun --account=nnXXXXk --time=00:01:00 --mem=1G --pty bash -i`

The parameters before `--pty bash -i` could also be used to submit a batch job
(often they are put into a shell script). On Abel you may have used the command
`qlogin` to submit an interactive job.

Our documentation provides additional information about [interactive jobs](jobs/interactivejobs.md).
You may also check the manual page for srun with the command `man srun`.

**Exercise 5:** Determine the compute node on which the job runs and list all modules
which are loaded once the job has started.

**Exercise 6:** Play with the srun command changing the wall time limit (`--time`)
and specifying that it should run on three nodes (parameter `--nodes`).

**Exercise 7:** What happens to your interactive job if your connection to Saga
terminates? Does the job continue to run (as a batch job would do) or is it
terminating (as a shell would do)?

## Installing software

## Batch jobs

## Transferring files to Saga

## Backing up data