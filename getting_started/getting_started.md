# Getting started

This page is meant to get you started on our resources and briefly list the
essentials.  In the menu on the left you will then find more in-depth
documentation on these topics.


## Getting access

To get access you need two things:
- {ref}`apply for a user account <applying-account>`
- {ref}`compute/storage resource allocation <applying-computing-storage>`


## Information on available CPU hours and disk space

This will list your available projects and the remaining CPU hours:
```console
$ cost
```

This will give you information about your disk usage:
```console
$ dusage
```


## Logging in

Logging into the machines involves the use of Secure Shell (SSH) protocol,
either in a terminal shell or through a graphical tool using this protocol
under the hood.  SSH login is available natively to Linux or macOS. Also on
Windows a number of good tools for this exists.

For more information and examples see {ref}`ssh`.

```console
$ ssh <username>@<machinename>
```

Replace `<username>` with your registered username and `<machinename>` with the 
specific machine name.

The machine names are:
- `betzy.sigma2.no` - {ref}`betzy`
- `fram.sigma2.no` - {ref}`fram`
- `saga.sigma2.no` - {ref}`saga`
- `login.nird.sigma2.no` - {ref}`nird`


## Transfering files

To copy files from your machine to the HPC machines, use SSH file transfer
Protocol (SFTP) or Secure Copy (SCP). They are available as command-line tools
for Linux and MacOS X but Windows users need to download a separate SCP or FTP
client, such as [WinSCP](https://winscp.net/) or [MobaXterm](https://mobaxterm.mobatek.net/).

For example, to copy `projectfiles.tar.gz` to the home directory of myusername to
Fram, type (the colon at the end is important):

```console
$ scp projectfiles.tar.gz myusername@fram.sigma2.no:
```

For more information please see our page on {ref}`file-transfer`.


## Remote desktop

The Fram and Saga systems provide a {ref}`remote-desktop` service.

**Quickstart**: Use a VNC client to log into `desktop.fram.sigma2.no:5901` or
`desktop.saga.sigma2.no:5901`. A web based remote desktop service is also
available <https://desktop.fram.sigma2.no:6080>. Access to these services are
blocked outside the Norwegian research network, e.g. only accessible from
UNINETT and partner institutions.


## Modules

To keep track of the large number of different pieces of software that is
typically available on a shared HPC cluster, we use something called a software
module system. This allows us to have many different versions of compilers,
libraries, and applications available for different users at the same time
without conflicting each other.

By default when you log in to the cluster you will get a clean environment with
nothing but standard system compilers and libraries. In order to make your
favourite software application available to you, you need to load its module
into your environment, which is done using the `module` command

```console
$ module <options> <modulename>
```

Some of the more common options include:

* `avail` - list the available modules
* `list` - list the currently loaded modules
* `load <modulename>` - load the module called `modulename`
* `unload <modulename>` - unload the module called `modulename`
* `show <modulename>` - display configuration settings for `modulename`

For more details please see {ref}`module-scheme`.


## Running applications

The HPC machines provide compute nodes for executing applications. To ensure
fair access to the resources, the HPC machines run applications as _jobs_ in a
_queue system_, which schedules the tasks and process to run on compute nodes.
All systems use the Slurm queue system.

A job is described by a _batch script_, which is a shell script (a text file)
with `SBATCH` options to specify the needed resources and commands to perform
the calculations. All batch scripts must contain _at least_ the following
two `SBATCH` options (on {ref}`saga` you also need to indicate maximum memory):

```
#!/bin/bash -l

# account name
#SBATCH --account=nnXXXXk

# max running time in d-hh:mm:ss
# this helps the scheduler to assess priorities and tasks
#SBATCH --time=0-00:05:00
```

For more details please see {ref}`job-types-and-job-scripts`.
