(module-scheme)=

# Software module scheme

Since a HPC cluster is shared among many users, and also holds a significant size in contrast to most desktop compute machinery around, the amount of installed software spans many applications in many different versions and quite a few of them are installed typically non-standard places for easier maintenance (for admin crew), practical and security reasons. It is not possible (nor desirable) to use them all at the same time, since different versions of the same application may conflict with each other. Therefore, it is practical to provide the production environment for a given application outside of the application itself. This is done using a set of instructions and variable settings that are specific for the given application called an application module. This also simplifies control of which application versions are available in a specific session.

The main command for using this system is the `module` command. You can find a list of all its options by typing:

	module --help

In NRIS we use the Lmod module system; for more info see <https://lmod.readthedocs.io/en/latest/>. The table below list the most commonly used options.

## Command summary

| Command     | Description     |
| :------------- | :------------- |
| `module overview`     | List all software packages installed with a count of each module |
| `module available`    | List all available modules and extensions |
| `module --nx available`    | List all available modules but showing extensions |
| `module --show_hidden available`     | List all available modules, including hidden modules |
| `module spider <software>`     | Search for a `<software>` among installed modules, including extensions |
| `module load <module>`     | Load a `<module>` |
| `module list`     | List modules that are currently loaded |
| `module reset`     | Unload all loaded modules and reset modules to system default |
| `module swap <module1> <module2>`     | LReplace `<module1>` with `<module2>` |
| `module show <module>`     | Show the commands in the `<module>` file |
| `module save <name>`     | Save the current list of modules to `<name>` collection |
| `module savelist`     | List all saved collections |
| `module restore <name>`     | Restore modules from `<name>` collection |
| `module use [-a] <path>`     | Prepend or append `<path>` to MODULEPATH |


We are using self-contained modules in NRIS, meaning that a given module loads all dependecies necessary. It is recommended to `module list` after loading a set of modules to check that the correct environment is set up. It is also good practice to start all job scripts with a `module reset` before loading all necessary modules for the calculation. This will prevent including any loaded modules in the current login session.

```{note}
The `module reset` command will inform you that some modules (like `StdEnv`)
were not unloaded. Such modules are made "sticky" because they are necessary
for the system to work, and they should not be `--force` purged as the message
suggest. If this warning message annoys you, you can suppress it with the `--quiet`
option instead.
```

Some modules are mainly intended as dependencies for others, and are typically
not very useful by themselves. Such modules are made hidden to the `module available`
command to avoid cluttering the listed output. However, if you are compiling
your own code some of these might still be useful, and you can still load them.
To include hidden modules you can add the `--show-hidden` option to the `module
available` search.


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


## Tutorial on module system for software
[Introduction to HPC - Accessing software](https://training.pages.sigma2.no/tutorials/hpc-intro/episodes/14-modules.html)
