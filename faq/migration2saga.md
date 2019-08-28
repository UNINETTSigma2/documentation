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

You may notice that the name/version identifiers for modules on Saga differ from
other systems, particularly when you're used to the module system on Abel.
On Saga (and Fram) the *version*
(part following the `/`) decodes both the version of the software package and the
tool chain (decodes a compiler version and supporting libraries) used to build the package.
The structure of these identifiers is the same for Saga and Fram. While the modules
system on Abel is using similar concept (name/version) as identifier, the *tool chain*
part usually contains only a brief keyword for the compiler being used. As an example,


## Interactive jobs

## Installing software

## Batch jobs

## Transferring files to Saga

## Backing up data