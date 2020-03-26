# Installed Software

The `module` command is for loading or listing available modules.

```
module [options] [module name]
```

To view the full list of options, enter `man module` in the command line. Here is a brief list of common module options:

* _avail_ - list the available modules
* _list_ - list the currently loaded modules
* _load  <module name>_ - load the module called modulename
* _unload  <module name>_ - unload the module called module name
* _show <module name>_  - display dependencies and environment variables
* _spider <module name>_  - print module description

For example, to display all available modules and load the Intel toolchain on Fram, enter:

```
module avail
module load intel/2017a
```

Modules may load other modules as part of its dependency. For example, loading the Intel version loads related modules to satisfy the module's dependency.
The `module show` command displays the other modules loaded by a module. The `module spider` command displays the module's description.

For installed SW on:

* Fram: See [List of installed software on Fram](installed_software/fram_modules.md)
* Saga: See [List of installed software on Saga](installed_software/saga_modules.md)
* Betzy (currently none, but will come eventually ;-))
