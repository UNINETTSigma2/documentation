# Using shared resources responsibly

One of the major differences between using remote HPC resources and your own
system (e.g. your laptop) is that **HPC resources are shared**.  Please use the
shared resources responsibly. Below we list few things to think about for a
more responsible resource use.

Please also make sure that you have gone through the documentation about
{ref}`job-types`, {ref}`queue-system`, {ref}`hardware-overview`,
{ref}`choosing-memory-settings`, and {ref}`choosing-number-of-cores`, to verify
that you are submitting the right job to the right partition to the right
hardware and not wasting resources. 


## Be kind to the login nodes and other users

The login node is often busy managing all the logged-in users, creating and
editing files and compiling software. If the machine runs out of memory or
processing capacity, it will become very slow and unusable for everyone.

**Always use the queue system for running jobs**. The login nodes are only for
file transfer, compilation, editing, job submission and short tests, etc. If
you run production jobs on the login nodes, we will need to stop them and email
you about it. More likely, another frustrated user might email us first and
complain about the too slow login node.

**Don't run interactive calculations on the login nodes**. If you need to run a job
interactively (not scheduled), have a look at {ref}`interactive-jobs`.


## Adjust required memory, number of cores, and time to what your job really needs

Do not ask for a lot more memory or number of cores or time than you need.
This may unnecessarily deplete your quota and may also delay the start of your
calculations.  It may also delay the start of calculations for others and
deplete available resources for others.

Please read these before asking for a lot "just to be on the safe side":
- {ref}`choosing-memory-settings`
- {ref}`choosing-number-of-cores`

Don't use `--exclusive` in job scripts unless explicitly told by NRIS
staff to do so.  This is especially important if you use `--mem-per-cpu`.


## Have a backup plan

See the documentation about {ref}`storage-backup` to learn what folders are
backed up and how.

However, **your data is your responsibility**.  Make sure you understand what
the backup policy is on the file systems on the system you are using and what
implications this has for your work if you lose your data on the system.

Make sure you have a robust system in place for taking copies of critical data
off the HPC system wherever possible to backed-up storage. Tools such as
`rsync` can be very useful for this.

Your access to the shared HPC system will generally be time-limited, so you
should ensure you have a plan for transferring your data off the system before
your access finishes. The time required to transfer large amounts of data
should not be underestimated, and you should ensure you have planned for this
early enough (ideally, before you even start using the system for your
research).


## Transferring data

Disk speed, meta-data performance, network speed, and firewall speed may limit
the transfer bandwidth.

Here are tips to make your data transfer easier and faster:

**Plan for it**: If you need to transfer large amount of data, don't start on
the last day of your project. Data transfer may take hours or even days.

Please read our page about {ref}`file-transfer`.
