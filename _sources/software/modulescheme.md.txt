(module-scheme)=

# Software module scheme

Since a HPC cluster is shared among many users, and also holds a significant size in contrast to most desktop compute machinery around, the amount of installed software spans many applications in many different versions and quite a few of them are installed typically non-standard places for easier maintenance (for admin crew), practical and security reasons. It is not possible (nor desirable) to use them all at the same time, since different versions of the same application may conflict with each other. Therefore, it is practical to provide the production environment for a given application outside of the application itself. This is done using a set of instructions and variable settings that are specific for the given application called an application module. This also simplifies control of which application versions are available in a specific session.

```{admonition} Important information
For the software module scheme on the Olivia cluster, please refer to the page {ref}`olivia-software`.
```

The main command for using this system is the module command. You can find a list of all its options by typing:

	module --help

We use the lmod module system; for more info see <https://lmod.readthedocs.io/en/latest/> in NRIS currently. Below we listed the most commonly used options, but also feel free to investigate options in this toolset more thoroughly on developers site.


## Which modules are currently loaded?

To see the modules currently active in your session, use the command:

	module list


## Which modules are available?

In order to see a complete list of available modules, issue the command:

	module avail

The resulting list will contain module names conforming to the following pattern:

* name of the module
* /
* version

The `avail` option can also be used to search for specific software, e.g.

	module avail netcdf

will list all modules matching the string "netcdf" (case insensitive).

```{note}
Some modules are mainly intended as dependencies for others, and are typically
not very useful by themselves. Such modules are made hidden to the `module avail`
command to avoid cluttering the listed output. However, if you are compiling
your own code some of these might still be useful, and you can still load them.
To include hidden modules you can add the `--show-hidden` option to the `module
avail` search.
```

## Which extensions are available?

Modules can contain extensions (**Perl modules**, **Python packages** and **R packages**). 
Extensions are marked with an `(E)` and can not be loaded directly. You will have to find and load the module that contains the extension.

List all versions of the Python package `numpy` that are installed:

```
module spider numpy
```

Five different version of numpy is installed in this example:
```
     Versions:
        numpy/1.22.3 (E)
        numpy/1.24.2 (E)
        numpy/1.25.1 (E)
        numpy/1.26.2 (E)
        numpy/1.26.4 (E)

```

Now list the module that contains a specific version of `numpy`:

```
module spider numpy/1.26.4
```

In this case it was the `SciPy-bundle/2024.05-gfbf-2024a` module that provides the Python package `numpy` version `1.26.4`. In order to see what other extensions are included in the module, you can run:

```
module spider SciPy-bundle/2024.05-gfbf-2024a
```

We see that these Python packages are included in the module:
```

      Included extensions
      ===================
      beniget-0.4.1, Bottleneck-1.3.8, deap-1.4.1, gast-0.5.4, mpmath-1.3.0,
      numexpr-2.10.0, numpy-1.26.4, pandas-2.2.2, ply-3.11, pythran-0.16.1,
      scipy-1.13.1, tzdata-2024.1, versioneer-0.29
```

```{note}
If you do not want to see extensions in the output you can run the module command with the `--nx` option, like this: `module --nx avail`
```

## How to load a module

In order to make, for instance, the NetCDF library available, get the full list of available netCDF modules first by typing: 

	module avail netCDF

Pick up one from the list (for example netCDF/4.9.2-gompi-2023a) and then issue the command:

	module load netCDF/4.9.2-gompi-2023a

Note that we currently do **not have** default modules on NRIS machines, so you need to write full module name when loading!


## How to unload a module

Keeping with the above example, use the following command to unload the NetCDF module again:

	module unload netCDF

Note that this will only unload the loaded module with "netCDF/"-namebase, in this case the module named netCDF/4.4.1.1-intel-2018a-HDF5-1.8.19. To unload everything you can type

	module reset

```{note}
The `module reset` command will inform you that some modules (like `StdEnv`)
were not unloaded. Such modules are made "sticky" because they are necessary
for the system to work, and they should not be `--force` purged as the message
suggest. If this warning message annoys you, you can suppress it with the `--quiet`
option instead.
```


## How to switch to a different version of a module

Switching to another version is similar to loading a specific version. As an example, if you want to switch from the current loaded netCDF to an older one; netCDF/4.9.0-gompi-2022b:

	module switch netCDF/4.9.2-gompi-2023a netCDF/4.9.0-gompi-2022b

This, more compact syntax will fortunately also work:

	module switch netCDF netCDF/4.9.0-gompi-2022b

```{note}
We are using self-contained modules in NRIS, meaning that a given module
loads all dependecies necessary. This is in slight contrast to old policies and also means it is possible
to make a mess if you load extra modules in job scripts after loading the main software module.
We recommend doing `module list` after every load (to inspect) and unloading any
conflicting packages, if possible. It is also good practice to start all job scripts
with a `module reset`, before loading all necessary modules for the calculation.
```


## How to save and restore your module environment

When you have loaded all necessary modules for a particular purpose and made sure that
your environment is working correctly, you can save it with

	module save <name-of-env>

and later restore it with

	module restore <name-of-env>

To list all your saved environments

	module savelist

This feature is particularly convenient if you spend a lot of time compiling/debugging
in interactive sessions. For production calculations using job scripts it is still
recommended to load each module explicitly for clarity.


## GPU modules 

### Saga
There are two types of GPU nodes on Saga, located in two distinct SLURM partitions:

* Intel CPU  with 4X Tesla P100, 16GB, `--partition=accel`
* AMD CPUs with 4XA100, 80GB, `--partition=a100`

These are different architectures. By default, Saga loads the Intel software environment,
If you want to run/compile software for the nodes with the AMD CPUs and A100 GPUS 
you need to get an allocation on the `a100` partition. Then, inside the allocation,
or inside your job script, switch the module environment

```
module --force swap StdEnv Zen2Env

```  
Note that installed modules can vary between the two node types.


## Tutorial on module system for software
[Introduction to HPC - Accessing software](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html)
