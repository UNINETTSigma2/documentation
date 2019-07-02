# Saga

*Note! Saga is scheduled to enter full production during fall 2019.*

**[Info for pilot users below](#primer-for-pilot-users)**

In norse mythology [Saga](https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr) is the goddess associated with wisdom.
The new Linux cluster hosted at [Norwegian University of Science and Technology](https://www.ntnu.edu)
(NTNU) is a shared resource for research computing capable of 645 TFLOP/s
theoretical peak performance. It is scheduled to enter full production during fall 2019.

Saga is a distributed memory system which consists of 244 dual/quad socket nodes,
interconnected with a high-bandwidth low-latency InfiniBand
network. Saga contains 200 standard, 28 mediummem, 8 largemem and 8 gpu compute nodes.
Each standard (or mediummem) compute node has two 20-core Intel Skylake
chips (2.0 GHz), 192 GiB memory (mediummem: 384 GiB) and a fast local NVMe disk (400 GB).
Each largemem compute node
contains four 16-core Intel Skylake chips (2.1 GHz) and 3072 GiB memory. Each GPU node
contains two 12-core Intel Skylake chips (2.6 GHz), 384 GiB memory, 4 NVIDIA P100 GPUs
and fast local SSD-based disk RAID (about 8 TB in RAID1).

The total number of compute cores is 9824. Total memory is 75 TiB.


| Details     | Saga     |
| :------------- | :------------- |
| System     |Hewlett Packard Enterprise - Apollo 2000/6500 Gen10  |
| Number of Cores     |	9824  |
| Number of nodes     |	244  |
| CPU type     |	Intel Xeon-Gold 6138 2.0 GHz (standard & mediummem)<br> Intel Xeon-Gold 6130 2.1 GHz (largemem)<br> Intel Xeon-Gold 6126 2.6 GHz (gpu)  |
| Max Floating point performance, double     |	645 Teraflop/s  |
| Total memory     |	75 TiB  |
| Total NVMe+SSD local disc | 89 TiB + 60 TiB |
| Total number of GPUs | 32 NVIDIA P100, 16 GiB RAM |
| Total disc capacity     |	1 PB  |


More information on how to use Saga
will be added here when the system becomes ready for production.

## Primer for pilot users
In general, the user environment on Saga is designed to be as similar as possible
to the one on Fram. Users coming from Abel will need to adopt a bit to the different
queue system setup and to the newer software module system.

Below are key information listed and links to existing
documentation for Fram provided.

### Getting support
Please contact support via `support@metacenter.no` and clearly mention that you
are a pilot user on Saga.

### Access
Login to Saga with your Notur account and password. The login machine's name is `saga.sigma2.no`. For example, using `ssh` do

`ssh YOUR_USERNAME@saga.sigma2.no`

### File systems
Saga has one parallel file system mounted under `/cluster`. While its layout is
identical to that on [Fram](https://documentation.sigma2.no/storage/storagesystems.html), it is based on BeeGFS
(vs Lustre on Fram), hence has slightly different features, and was designed to
handle I/O-intensive workloads.

### Scientific software
Saga uses `lmod` and `EasyBuild`, so commands to work with modules are identical.
At the start, Saga may have only limited number of modules installed. Most useful
commands are `module avail` (list available modules), `module load <MODULE>`(to
load a module), `module list` (to list currently loaded modules) and
`module purge` (to unload all modules). For more details, please
see documentation about [software modules on Fram](https://documentation.sigma2.no/apps/modulescheme.html).

### Queue system
Saga uses Slurm. For users coming from Fram, Abel or Stallo, basic commands are
the same or similar, however the configuration (partitions, QoS, limits, etc.) differs
from the other systems. For general information, see our documentation on the
[queue system](https://documentation.sigma2.no/jobs/framqueuesystem.html), [job types](https://documentation.sigma2.no/jobs/jobtypes.html) and
[job scripts](https://documentation.sigma2.no/jobs/jobscripts.html).

#### Normal jobs
This is the default type of job.  These jobs run on the ordinary
compute nodes, which have 40 cpus (cores) and ~ 185 GiB RAM.  A subset
of the nodes have ~ 370 GiB RAM.

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
Saga has 8 `bigmem` nodes with 24 cpus and ~ 3 TiB RAM.

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

Se "Normal jobs" above for description of `--ntasks-per-node`,
`--nodes` and `--cpus-per-task`.

#### GPU jobs
Saga has 8 nodes with 24 cpus, ~ 185 GiB RAM and 4 GPU cards.

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

Se "Normal jobs" above for description of `--ntasks-per-node`,
`--nodes` and `--cpus-per-task`.
