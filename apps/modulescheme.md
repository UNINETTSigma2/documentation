# Software Module Scheme

Since a HPC cluster is shared among many users, and also holds a significant size in contrast to most desktop compute machinery around, the amount of installed software spans many applications in many different versions and quite a few of them are installed typically non-standard places for easier maintenance (for admin crew), practical and security reasons. It is not possible (nor desirable) to use them all at the same time, since different versions of the same application may conflict with each other. Therefore, it is practical to provide the production environment for a given application outside of the application itself. This is done using a set of instructions and variable settings that are specific for the given application called an application module. This also simplifies control of which application versions are available in a specific session.

The main command for using this system is the module command. You can find a list of all its options by typing:

	module --help

We use the lmod module system; for more info see <https://lmod.readthedocs.io/en/latest/> in the Metacenter currently. Below we listed the most commonly used options, but also feel free to ivestigate options in this toolset more thoroughly on developers site.

## Which modules are currently loaded?
To see the modules currently active in your session, use the command:

	module list

### Which modules are available?
In order to see a complete list of available modules, issue the command:

	module avail

The resulting list will contain module names conforming to the following pattern:

* name of the module
* /
* version

## How to load a module
In order to make, for instance, the NetCDF library available issue the command:

	module load netCDF/4.4.1.1-intel-2018a-HDF5-1.8.19

Note that we currently do **not have** default modules on Metacenter machines, so you need to write full module name when loading! 

### How to unload a module
Keeping with the above example, use the following command to unload the NetCDF module again:

	module unload netCDF

Note that this will only unload the loaded module with "netCDF/"-namebase, in this case the module named netCDF/4.4.1.1-intel-2018a-HDF5-1.8.19. To unload everything you can type

	module purge

### How do I switch to a different version of a module?
Switching to another version is similar to loading a specific version. As an example, if you want to switch from the current loaded netCDF to an older one; netCDF/4.4.0-intel-2016a:

	module switch netCDF/4.4.1.1-intel-2018a-HDF5-1.8.19 netCDF/4.4.0-intel-2016a

This, more compact syntax will fortunately also work:

	module switch netCDF netCDF/4.4.0-intel-2016a

**Beware: We are using self-contained modules in the Metacenter**, meaning that a given module loads all dependecies necessary. This is in slight contrast to old policies on for instance Stallo, and also means that you might mess upp quite significantly if you load extra modules in job scripts after loading the main software module. 

We recommend doing *module list* after every load (to inspect) and unloading any conflicting packages, if possible. 
