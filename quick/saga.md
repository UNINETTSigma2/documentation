# Saga

*Note! Saga is scheduled to enter full production during fall 2019.*

In norse mythology [Saga](https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr) is the goddess associated with wisdom.
The new Linux cluster hosted at [Norwegian University of Science and Technology](https://www.ntnu.edu)
(NTNU) is a shared resource for research computing capable of 645 TFLOP/s
theoretical peak performance. It is scheduled to enter full production during fall 2019.

Saga is a distributed memory system which consists of 244 dual/quad socket nodes,
interconnected with a high-bandwidth low-latency InfiniBand
network. Saga contains 200 `normal`, 36 `bigmem` and 8 `accel` compute nodes.
Each normal compute node has two 20-core Intel Skylake
chips (2.0 GHz), 192 GiB memory (186 GiB available for jobs) and a fast local NVMe disk (ca 300 GB shared by jobs).
There are two kinds of bigmem compute nodes: 28 contain two 20-core Intel Skylake chips (2.0 GHz), 384 GiB memory
(377 GiB available for jobs) and a fast local NVMe disk (ca 300 GB shared by jobs); 8 contain four 16-core Intel
Skylake chips (2.1 GHz) and 3072 GiB memory (3021 GiB available for jobs). Each GPU node
contains two 12-core Intel Skylake chips (2.6 GHz), 384 GiB memory (377 GiB available for jobs), 4 NVIDIA P100 GPUs
and fast local SSD-based disk RAID (about 8 TB in RAID1).

The total number of compute cores is 9824. Total memory is 75 TiB.


| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	9824  |
| Number of nodes     |	244  |
| Number of GPUs | 32 |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz (normal)<br> Intel Xeon-Gold 6130 2.1 GHz (bigmem)<br> Intel Xeon-Gold 6126 2.6 GHz (accel)  |
| GPU type     |    NVIDIA P100, 16 GiB RAM (accel) |
| Total max floating point performance, double     |	645 Teraflop/s (CPUs) + 150 Teraflop/s (GPUs) |
| Total memory     |	75 TiB  |
| Total NVMe+SSD local disc | 89 TiB + 60 TiB |
| Total parallel filesystem capacity     |	1 PB  |

More information on how to use Saga
will be included in the existing sections, e.g,
SOFTWARE, CODE DEVELOPMENT, JOBS, FILES AND STORAGE, etc.

## Primer for early users
In general, the user environment on Saga is designed to be as similar as possible
to the one on Fram. Users coming from Abel will need to adopt a bit to the different
queue system setup and to the newer software module system.

Below are key information listed and links to existing
documentation for Fram provided.

### Important !
Currently, there is NO BACKUP of anything on Saga, e.g., not for $HOME folders, project
folders, shared folders. Make sure to make copies of precious scripts, setups, and data.
We hope to have backup available soon.

Note, while [**defaults for quota**](#file-systems-and-quota) have been defined,
these are not yet enforced. Make sure to not use more than the quota or you will
run into problems once we are going to enforce them.

### Getting support
Please, contact support via `support@metacenter.no` and clearly mention that you
are a user on Saga.

Please be as specific as possible with your support requests,
i.e., include

* job ids,
* job scripts (path to a specific script, not just a directory containing lots of scripts),
* job output files (by default named `slurm-JOBID.out` and stored in the directory from where you submitted the job)
* command used to submit a job and the directory from where you submitted a job,
* modules loaded (use `module list` to show them),
* terminal sequences (commands you used + outputs they produced),
* paths to data used, *and*
* any other information that you think could be useful (*same script worked last week*, *same script works for my colleague*, etc.)

### Access
Login to Saga with your Notur account and password. The login machine's name is `saga.sigma2.no`. For example, using `ssh` do

`ssh YOUR_USERNAME@saga.sigma2.no`

### File systems and quota
Saga has one parallel file system mounted under `/cluster`. While its layout is
identical to that on [Fram](../storage/storagesystems.md), it is based on BeeGFS
(vs Lustre on Fram), hence has slightly different features, and was designed to
handle I/O-intensive workloads.

**Default quota on $HOME folders is 20 GiB.** Quota for project folders are according
to the grant you have received (based on what you asked for in your application for
compute time). Quota for shared folders is according to agreements by several projects
and Sigma2.

### Scientific software
Saga uses `lmod` and `EasyBuild`, so commands to work with modules are identical.
At the start, Saga may have only limited number of modules installed. Most useful
commands are `module avail` (list available modules), `module load <MODULE>`(to
load a module), `module list` (to list currently loaded modules) and
`module purge` (to unload all modules). For more details, please
see documentation about [software modules on Fram](../apps/modulescheme.md).

### Queue system
Saga uses Slurm. For users coming from Fram, Abel or Stallo, basic commands are
the same or similar, however the configuration (partitions, QoS, limits, etc.) differs
from the other systems. For general information, see our documentation on the
[queue system](../jobs/framqueuesystem.md), [job types](../jobs/jobtypes.md) and
[job scripts](../jobs/jobscripts.md).

#### Normal jobs
This is the default type of job.  These jobs run on the `normal`
compute nodes, which have 40 cpus (cores) and ~ 186 GiB RAM.

A note for users from Fram: on Saga, the queue system only hands out cpus
and memory, not whole node, so one _must_ specify `--mem-per-cpu` (or
`--mem`) for all types of jobs.

Example job script:

    #!/bin/bash
    #SBATCH --account=nn9999k
	#SBATCH --time=24:00:00    # Max walltime is 7 days.
	#SBATCH --mem-per-cpu=4G
	#SBATCH --ntasks=16        # Default is 1
	
	set -o errexit  # Recommended for easier debugging
    
	## Load your modules
	module purge   # Recommended for reproducibility
	module load somemodule/version
    
	## Optionally, copy files to $SCRATCH or $USERWORK
    
	YourCommand

Instead of `--ntasks`, one can use `--ntasks-per-node` with `--nodes`
to select how many tasks should go on each node.

For multithreaded jobs, use `--cpus-per-task` to select the number of
threads per task.  This will set the `$OMP_NUM_THREADS` environment
variable, which OpenMP programs use.

#### Bigmem jobs
Saga has 36 `bigmem` nodes in two configurations. Twenty eight (28) nodes
have 40 cpus (cores) and ~ 377 GiB RAM. Eight (8) nodes have 64 cpus and
~ 3 TiB RAM.

Example job script:

    #!/bin/bash
    #SBATCH --account=nn9999k
	#SBATCH --partition=bigmem   # To use the bigmem nodes
	#SBATCH --time=24:00:00      # Max walltime is 14 days.
	#SBATCH --mem-per-cpu=100G
	#SBATCH --ntasks=4
	
	set -o errexit  # Recommended for easier debugging
    
	## Load your modules
	module purge   # Recommended for reproducibility
	module load somemodule/version
    
	## Optionally, copy files to $SCRATCH or $USERWORK
    
	YourCommand

See [Normal jobs](#normal-jobs) above for description of `--ntasks-per-node`,
`--nodes` and `--cpus-per-task`.

#### GPU jobs
Saga has 8 nodes with 24 cpus, ~ 377 GiB RAM and 4 GPU cards.

Example job script:

    #!/bin/bash
    #SBATCH --account=nn9999k
	#SBATCH --partition=accel    # To use the accelerator nodes
	#SBATCH --gres=gpu:1         # To specify how many GPUs to use
	#SBATCH --time=24:00:00      # Max walltime is 14 days.
	#SBATCH --mem-per-cpu=10G
	#SBATCH --ntasks=1
	
	set -o errexit  # Recommended for easier debugging
    
	## Load your modules
	module purge   # Recommended for reproducibility
	module load somemodule/version
    
	## Optionally, copy files to $SCRATCH or $USERWORK
    
	YourCommand

Use `--gres=gpu:N` to select how many GPU cards the job will get (N = 1,
2, 3 or 4).

See [Normal jobs](#normal-jobs) above for description of `--ntasks-per-node`,
`--nodes` and `--cpus-per-task`.
