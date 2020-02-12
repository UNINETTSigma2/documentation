# Getting Started

This page lists information about how to get started with running applications
on the HPC machines.

## Getting Access

To gain access to the HPC machines, a formal application is needed. The process
is explained at our page for [how to apply for user account](https://www.sigma2.no/how-apply-user-account).

## Logging In

Logging into the machine involves the use of Secure Shell (SSH) in a terminal.
SSH login is available natively to Linux or Mac OSX (or any UNIX-based systems).
On Windows machines an SSH client is needed, for example, [PuTTY](http://putty.org).

```shell
    ssh <username>@<machinename>
```

Replace `<username>` with your registered username and `<machinename>` with the specific machine name.

The machine names are:

* `fram.sigma2.no` - Fram
* `saga.sigma2.no` - Saga
* `login.nird.sigma2.no` - NIRD.  See [NIRD](../storage/nird.md) for
  more about NIRD.

First time you log in, and everytime there is a change in the
*ssh-server*, OS or hardware on the system you log in to, you will be asked to confirm the ssh-server fingerprint. Typical message is:

    The authenticity of host 'fram.sigma2.no (158.39.114.72)' can't be established.
    
The procedure then is to check the stated `ECDSA key fingerprint` with the one printed here: [ssh SHA256 fingerprint](../faq/ssh.md#sha256-fingerprint).

## Remote desktop

The Fram, Saga and Stallo systems provides a remote desktop service. [See here for tutorial and details.](remote-desktop.md)

**Quickstart**: Use a VNC client to log into `desktop.fram.sigma2.no:5901` or `desktop.saga.sigma2.no:5901` (for information on Stallo see here <http://stallo-gui.uit.no>). A web based remote desktop service is also available <https://desktop.fram.sigma2.no:6080>. Access to these services are blocked outside the Norwegian Research network, e.g. only accessible from UNINETT and partner institutions. (workarounds are described in the [tutorial](remote-desktop.md))

## Development environment

Abel, Stallo, Fram and Saga run CentOS Linux distributions as operating system. The machines can run C/C++ or Fortran OpenMP and MPI applications, and depending on the machine, various open-source and third party applications. The Programming Environment page has more information about third-party applications.

### Unix CLI

It is expected that the user is familiar with command-line interfaces (CLIs), but for those who are not familiar with commands, the UiB and UiT pages have several tutorials on the CLI environment:

* [Introduction to Unix CLI](https://docs.hpc.uib.no/wiki/Introduction_to_Unix_CLI) - UiB

To copy files from your machine to the HPC machines, use SSH File Transfer Protocol (SFTP) or Secure Copy (SCP). They are available as command-line tools for Linux and MacOS X but Windows users need to download a separate SCP or FTP client, such as WinSCP.

For example, to copy projectfiles.tar.gz to the home directory of myusername to Fram, type:

    scp projectfiles.tar.gz myusername@fram.sigma2.no:

The [Transferring Files](../faq/file_transfer.md) page has more information about transferring files to Fram.

### Modules

Modules enable applications to run different environment configurations. For example, an application may run with either the Intel compiler or the GNU compiler by loading different modules. The module command is for loading or listing available modules.

    module [options] [module name]

To view the full list of options, enter man module in the command line. Here is a brief list of common module options:

* `avail` - list the available modules
* `list` - list the currently loaded modules
* `load  <modulename>` - load the module called `modulename`
* `unload  <modulename>` - unload the module called `modulename`
* `show <modulename>`  - display configuration settings for `modulename`

For example, to load the Intel toolchain on Fram, enter:

    module load intel/2017a

## Running Applications

The HPC machines provide compute nodes for executing applications. To ensure
fair access to the resources, the HPC machines run applications as _jobs_ in a
_queue system_, which schedules the tasks and process to run on compute
nodes. Abel, Stallo, and Fram use the Slurm queue system.

A job is described by a _batch script_, which is a shell script (a text file)
with `SBATCH` options to specify the needed resources and commands to perform
the calculations.  All batch scripts must contain _at least_ the following
`SBATCH` options:

    #Project name
    #SBATCH --account=<project_name>

    #Max running time. The wall clock time helps the scheduler assess priorities and tasks.
    #SBATCH --time=<wall_clock_time>

To submit a job, use the `sbatch` command followed by the batch script's name:

    sbatch <scriptfile>

See the "JOBS" section in the menue for documentation about jobs and the queue system.

## Account information

Information on available CPU-hours in your accounts:

    cost

## More Information

The menue on the left list the pages with more information about developing and running applications.
