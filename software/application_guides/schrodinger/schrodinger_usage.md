---
orphan: true
---

# Using the Schrodinger suite

Load the desired Schrodinger suite:
* `module purge`
* `module load Schrodinger/2021-2-intel-2020b`

Now you can call any Schrodinger software by using the \$SCHRODINGER variable from your command line, for example
`$SCHRODINGER/glide`.

You can also launch maestro by typing the command `maestro`. We however would generally encourage our users
to limit the use of graphical user interfaces to as little as possible. If you for some reason need to use the 
maestro gui, you must log in to SAGA with X11 forwarding, e.g. `ssh -Y you@saga.sigma2.no`.

To the extent it is possible, we recommend preparing input files etc. using a local version of maestro and uploading
the files to SAGA (`scp -r input_files/ you@saga.sigma2.no:/some/existing/directory`). To create the input files needed
to run a job from the command line, set up the job in Maestro, choose Write from the Job Settings button menu 
![settings](figures/settings.png), and if needed, modify the files that are generated. Jobs can then be submitted from
the command line using the `$SCRODINGER` variable. For example:

`"${SCHRODINGER}/glide" glide-grid_1.in -OVERWRITE -NJOBS 20 -DRIVERHOST localhost -SUBHOST batch-small -TMPLAUNCHDIR`

The above command submits the pre-created input file glide-grid_1.in (ligand docking) and keeps the driver (job control) 
on localhost (`-DRIVERHOST localhost`), preventing it from occupying a node. The `-SUBHOST -batch-small` tells 
Schrodinger to you use the job settings and qargs defined in your local [schrodinger.hosts](schrodinger_hosts.md) file 
with entry name batch-small for the `-NJOBS 20` subjobs. Thus, the above command will send a total of 20 subjobs, each
using the qargs defined in batch-small. 

## The DRIVERHOST and SUBHOST
Schrodinger uses its own [job control facility](job_control.md) that sits on top of the SLURM queuing system. When 
submitting a job, schrodinger will use one job as a driver, to control the remaining jobs. Thus, if you submit a job 
directly from maestro, or with `"${SCHRODINGER}/glide" glide-grid_1.in -OVERWRITE -NJOBS -HOST batch-small`, the driver 
will also be submitted to the queue along with the 20 subjobs. The problem with this is that the driver may run out of 
walltime before the subjobs have even started. This will effectively kill all the subjobs. In order to avoid this, you 
must specify a `-DRIVERHOST` and a `-SUBHOST`. Setting the `-DRIVERHOST` to localhost will put the driver on of the login 
node and the actual jobs on the host specified by `-SUBHOST` (compute nodes). This combination will not allow any subjobs 
to run on the `-DRIVERHOST`, only the driver itself, which is good. 

**Conclusion:** Do not submit jobs on the cluster directly from maestro, but from the command line specifying 
`-DRIVERHOST` and `-SUBHOST`.

## Submitting you jobs from command line
`"${SCHRODINGER}/package" job.in -DRIVERHOST localhost -SUBHOST hostname`

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)
* [Hosts file keywords](host_file_settings.md)
* [Job control facility](job_control.md)
* [Tuning](tuning.md)
