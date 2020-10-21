# Migration to a Metacenter HPC machine

In general, the user environment on all Metacenter machines should be as similar as possible.
Thus, for users moving internally between machines run by the Metacenter, they only need to focus on the following:

* [Hardware differences](/hpc_machines/hardware_overview.md): number of CPU cores, memory size, GPU-availability, external access from compute nodes.
* Software differences: [Installed SW](/software/installed_software.md), type of file-system, access limitation rules to \$HOME, same or different file systems on \$TMP and \$HOME.
* Jobtype policy differences in Resource Management System (SLURM) on different machines.
* Storage options.

For users either being novel to HPC in general, or having experience from other clusters - either local/private or foreign setup, basically the same rules apply - one must try to identify the critical differences in what one is used to and then adapt behaviour accordingly. 

## Major steps in migrating to a Metacenter HPC machines

* Read this documentation.
* Get an [account](/getting_started/applying_account.md) and [project](/getting_started/applying_resources.md) quota.
* Become aware of differences in disk quota, module system, job types, running jobs, how to get help, file system policies.
* Transfer data, scripts etc from other machines to the new machine.
* Modify scripts & routines to match differences on the new machine.
* **Verify that your jobs run efficiently and produce the same results as on other systems!**
* Be patient with user support [(support@metacenter.no)](mailto:support@metacenter.no), but don't hesitate to ask questions!

## Read about the current machines operated by the Metacenter

* [Fram](/hpc_machines/fram.md)
* [Saga](/hpc_machines/saga.md)
* [Betzy](/hpc_machines/betzy.md)
