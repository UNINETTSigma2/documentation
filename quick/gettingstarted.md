#Getting Started

This page lists information about how to get started with running applications
on the HPC machines.

##Getting Access

To gain access to the HPC machines, a formal application is needed. The process
is explained at the [User Access](https://www.sigma2.no/node/36) page.

##Logging In

Logging into the machine involves the use of Secure Shell (SSH) in a terminal.
SSH login is available natively to Linux or Mac OSX (or any UNIX-based systems).
On Windows machines an SSH client is needed, for example, [PuTTY](http://putty.org).

```
ssh <username>@<machinename>
```

Replace _<username>_ with your registered username and _<machinename>_ with the specific machine name.

The machine names are:

* abel.uio.no   - Abel
* stallo.uit.no - Stallo
* fram.sigma2.no - Fram

##Development environment

Abel, Stallo and Fram runs CentOS Linux distributions as operating system. The machines can run C/C++ or Fortran OpenMP and MPI applications, and depending on the machine, various open-source and third party applications. The Programming Environment page has more information about third-party applications.

### Unix CLI

It is expected that the user is familiar with command-line interfaces (CLIs), but for those who are not familiar with commands, the UiB and UiT pages have several tutorials on the CLI environment:

* [Introduction to Unix CLI](https://docs.hpc.uib.no/wiki/Introduction_to_Unix_CLI) - UiB

To copy files from your machine to the HPC machines, use SSH File Transfer Protocol (SFTP) and Secure Copy (SCP). They are available as command-line tools for Linux and MacOS X but Windows users need to download a separate SCP or FTP client, such as WinSCP.

For example, to copy projectfiles.tar.gz to the home directory of myusername to Fram, type:

```
scp projectfiles.tar.gz myusername@fram.sigma2.no:
```

###Modules

Modules enable applications to run different environment configurations. For example, an application may run with either the Intel compiler or the GNU compiler by loading different modules. The module command is for loading or listing available modules.

```
module [options] [module name]
```

To view the full list of options, enter man module in the command line. Here is a brief list of common module options:

* _avail_ - list the available modules
* _list_ - list the currently loaded modules
* _load  <module name>_ - load the module called modulename
* _unload  <module name>_ - unload the module called module name
* _show <module name>_  - display configuration settings for module name

For example, to load the Intel toolchain on Fram, enter:

```
module load intel/2017a
```


##Processes as Jobs

The HPC machines provide compute nodes for executing applications. To ensure fair access to projects and to take advantage of compute node performance, the HPC machines use a queue system to delegate the compute nodes to a project. The HPC machines run applications as jobs in the queuing system which schedules the tasks and process to run on compute nodes. Abel, Stallo, and Fram use Simple Linux Utility for Resource Management (Slurm).

To submit a job, use the _sbatch_ command followed by the batch script.

```
sbatch <scriptfile>
```

All batch scripts must contain the following sbatch commands:

    ```
    #Project name as seen in the queue
    #SBATCH --account=<project_name>

    #Application running time. The wall clock time helps the scheduler assess priorities and tasks.
    #SBATCH --time=<wall_clock_time>

    #Memory usage per core
    #SBATCH --mem-per-cpu=<size_megabytes>
    ```

##Account information

Information on available CPU-hours in your accounts:

```
$ cost
```

##More Information

The menu on the left list the pages with more information about developing and running applications.
