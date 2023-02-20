(getting-started)=

# Getting started

This page is meant to get you started on our resources and briefly list the
essentials.  In the menu on the left you will then find more in-depth
documentation on these topics.


## First time on a supercomputer?

Please read the **GETTING STARTED** section (left sidebar). In the sidebar
overview you will also find technical details about the machines, instructions
for using installed software, for submitting jobs, storage, and code
development.

Please do not hesitate to write to support@nris.no if you find documentation
sections which are not clear enough or have suggestions for improvements. Such
a feedback is very important to us and will count.


## How to get the most out of your allocation

We want to support researchers in getting the most out of the
high-performance computing services. When supporting users, we see that
these problems are very frequent:

- **Reusing outdated scripts** from colleagues without adapting them to
  optimal parameters for the cluster at hand and thus leaving few cores
  idle. Please check at least how many cores there are on a particular
  cluster node.
- **Requesting too much memory** which leads to longer queuing and less
  resource usage. Please check {ref}`choosing-memory-settings`.
- **Requesting more cores than the application can effectively use** without
  studying the scaling of the application. You will get charged more than
  needed and others cannot run jobs. If others do this, your own jobs queue.
- **Submitting jobs to the wrong queue** and then queuing longer than
  needed. Please take some time to study the different {ref}`job-types`.

If you are unsure about these, please contact us via
support@nris.no and we will help you to use your allocated
resources more efficiently so that you get your research results faster.


## Getting access

To get access you need two things:
- {ref}`apply for a user account <applying-account>`
- {ref}`compute/storage resource allocation <applying-computing-storage>`


## Information on available CPU hours and disk space

This will list your available projects and the remaining CPU hours
(see also {ref}`projects-accounting`):
```console
$ cost
```

This will give you information about your disk {ref}`storage-quota`:
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

If you are unfamiliar with working with Unix or want to refresh the basics, you
may want to [learn using the shell](https://effective-shell.com/).


## Transferring files

To copy files from your machine to the HPC machines, use SSH file transfer
Protocol (SFTP) or Secure Copy (SCP). They are available as command-line tools
for Linux and MacOS X but Windows users need to download a separate SCP or FTP
client, such as [WinSCP](https://winscp.net/) or [MobaXterm](https://mobaxterm.mobatek.net/).

For example, to copy `projectfiles.tar.gz` from your local home directory
to the remote home directory of `myusername` on
Fram, type (the colon at the end is important):

```console
$ scp projectfiles.tar.gz myusername@fram.sigma2.no:
```

For more information please see our page on {ref}`file-transfer`.


## Remote desktop

The Fram and Saga systems provide a remote desktop service. For more information about this, see {ref}`remote-desktop`.

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

For more details please see {ref}`running-jobs`.


## Learning about the Linux command line

Learning the basics about the Linux command line (shell) navigation and shell
scripting will make your daily work easier and more efficient.

If you are new to command line, go through some Linux tutorials on the subject
first.  Here are some useful pages you can look into: [shell
novice](https://swcarpentry.github.io/shell-novice/) or [Effective
shell](https://effective-shell.com). O therwise consult your local IT resources
for help.

However, do not be afraid to contact support if you are not an expert or have
knowledge of this. We will try our best to help you.
