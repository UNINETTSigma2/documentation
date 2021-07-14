# Checkpointing Jobs

Checkpointing is the action of saving the state of a running process to a check point image file.
Users can utilize checkpointing to pickup a job where it left off due to failing resources (e.g. hardware,
software, exceeded time and memory resources) and continue running. 
Users are encouraged to use application level checkpointing, that means to investigate whether the software
tools they're using are capable of stopping and restarting where a job leaves off. If it is available, 
it is recommended to use the software built in tools for checkpointing.

## Checkpointing on our Clusters

[DMTCP](http://dmtcp.sourceforge.net) (Distributed MultiThreaded Checkpointing)is a checkpointing package for applications.
DMTCP Checkpoint/Restart allows one to transparently checkpoint to disk a distributed computation. It works under Linux, 
with no modifications to the Linux kernel nor to the application binaries. It can be used by users (no root privilege needed).
One can later restart from a checkpoint. DMTCP supports both sequential and multi-threaded applications and it provides support 
for SLURM resource manager. 
The DMTCP module is available in all our machines **(Saga, Fram, Betzy)** and it is enabled by typing 

```module load DMTCP/2.6.0-GCCcore-9.3.0``` 

  There are two steps involved after loading the DMTCP module.

- First is to launch your application using `dmptcp_launch` by running the following

```[user1@login-1.SAGA ~]$ dmtcp_launch --new-coordinator --rm --interval <interval_time_seconds> <your_command>```

where `--rm` option enables SLURM support, `<interval_time_seconds>` is the time in seconds between automatic checkpoints, 
and `<your_command>` is the actual command you want to run and checkpoint

`dmtcp_launch` creates few files that are used to resume the cancelled job, such as `ckpt_*.dmtcp` and `dmtcp_restart_script*.sh`.
 Unless otherwise stated (using `--ckptdir option`), these files are stored in the current working directory.
 
 More `dmtcp_launch` options can be found by using :

```dmtcp_launch --help```

- The second step of DMTCP is to restart the cancelled job. This can be done by doing 

```./dmtcp_restart_script.sh```

 **Sample example of how to use DMPTCP in your slurm script**

- First submit your job with dmptcp `generic_job.sh' 

```bash
#!/bin/bash

# Job name:
#SBATCH --job-name=YourJobname
# Project:
#SBATCH --account=nnXXXXk
# Wall time limit:
#SBATCH --time=DD-HH:MM:SS
# Other parameters:
#SBATCH ...
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
### Load DMPTCP module 
module load DMTCP/2.6.0-GCCcore-9.3.0
### Load your software module
module load SomeProgram/SomeVersion
module list
## Do some work: Running under dmptcp control 
dmtcp_launch --new-coordinator --rm --interval 3600 YourCommands 
```

In this example, DMTCP takes checkpoints every hour '(--interval 3600)'

- Second, restart the job: If the job is killed for various reasons, it can be restarted using the following submit file: `generic_job_dmptcp_restart.sh`
```
#!/bin/bash

# Job name:
#SBATCH --job-name=YourJobname
# Project:
#SBATCH --account=nnXXXXk
# Wall time limit:
#SBATCH --time=DD-HH:MM:SS
# Other parameters:
#SBATCH ...
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
### Load DMPTCP module
module load DMTCP/2.6.0-GCCcore-9.3.0
### Load your software module
module load SomeProgram/SomeVersion
module list
# Start DMTCP
dmtcp_coordinator --daemon --port 0 --port-file /tmp/port
export DMTCP_COORD_HOST=`hostname`
export DMTCP_COORD_PORT=$(</tmp/port)
# Restart job 
# The script below(dmtcp_restart_script.sh) is created automatically as part of the checkpointing process.
./dmtcp_restart_script.sh
```

## More Information

dmtcp_restart generates new ckpt_*.dmtcp and dmtcp_restart_script*.sh files.
Therefore, if the restarted job is also killed due to unavailable/exceeded resources,
you can resubmit the same job again without any changes in the submit file shown above.
We recommend the users to delete  old ckpt_*.dmtcp files. 
Note that there is no guarantee that every application can be checkpointed and restarted with DMTCP.
Users are recommended to see the [DMTCP documentation](http://dmtcp.sourceforge.net/) and 
[DMTCP supported apps](http://dmtcp.sourceforge.net/supportedApps.html#xwindow)for further read.
