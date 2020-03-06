# Migration to a Metacenter HPC machine

In general, the user environment on all Metacenter machines should be as similar as possible. Thus, for users moving internally between machines run by the Metacenter, they only need to focus on the following:

* Differences in hardware status; corecount, memory size, GPU-availability, external access from compute.
* Differences is software status: The list of installed SW (use module avail), type of file-system, access limitation rules to $HOME, same or different filesystem on $TMP and $HOME.
* Jobtype policy differences in Resource Managment System (currently used SLURM) on different machines. See menu bar links for info about this. 
* Storage options.

For users either being novel to HPC in general, or having experience from outher clusters - either local/private or foreign setup, basically the same rules apply - one must try to identify the critical differences in what one is used to and then adapt behaviour accordingly. 

In a clearer way, thee major steps in migrating to a Metacenter HPC macines:

* Read this documentation
* Getting an account and project quota.
*  Getting aware of differences (disk quota, module system, job types, running jobs, how to get help, sile system policies).
*  Transferring data, scripts etc from other machines to the new machine.
*  Modifying scripts & routines to match differences on the new machine.
*  **Verifying that your jobs run efficiently and produce the same results as on other systems!**
*  Be patient with user support [(support@metacenter.no)](mailto:support@metacenter.no), but don't hesitate to ask questions!

## Read about the current machines operated by the Metacenter:

* [Fram](https://www.sigma2.no/systems#framq)
* [Saga](https://www.sigma2.no/systems#saga)
* [Betzy](https://www.sigma2.no/systems#betzy)

## Further reading:
* [Getting started](/quick/gettingstarted.md)
* [Transferring files](/storage/file_transfer.md)
* [Queue system](/jobs/queue_system.md)
* [Software module scheme](/apps/modulescheme.md)

Also see additional links in the left side menu.