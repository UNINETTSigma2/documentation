# Porting Batch Scripts from PBS/TORQUE

Converting a PBS/TORQUE script files to Slurm is simple because most of the commands
have direct equivalents in Slurm. The shell commands, variables, and some of the module names
need to be changed but the application code such as compiling and copying of files
can remain the same.

This page lists some ways to convert batch scripts from PBS/TORQUE to Slurm.

## Shell Commands

Many PBS/TORQUE commands directly translate to a Slurm command. Here are some
of the PBS/TORQUE commands with their Slurm counterparts.

| Shell Commands     | PBS/TORQUE     | Slurm     |
| :------------- | :------------- | :------------- |
| Job submission |  	qsub <*filename*> |  	sbatch <*filename*> |
| Job deletion |  	qdel <*job_id*> |  	scancel <*job_id*> |
| Job status (by job) |  	qstat <*job_id*> |  	squeue --job <*job_id*> |
| Full job status (by job) |  	qstat -f <*job_id*> | scontrol show job <*job_id*> |
| Job status (by user) |  	qstat -u <*username*> |  	squeue --user=<*username*> |

## Environment variables

| Environment variables |  	PBS/Torque |	SLURM |
| :------------- | :------------- | :------------- |
| Job ID  | $PBS_JOBID  | $SLURM_JOBID |
| Submit Directory | $PBS_O_WORKDIR | $SLURM_SUBMIT_DIR |
| Node List | 	$PBS_NODEFILE | $SLURM_JOB_NODELIST |

## Options and Settings

These are options that may be placed in the batch script or passed as arguments
to <*sbatch*.

| Options | PBS/Torque | SLURM |
| :------------- | :------------- | :------------- |
| Script directive | #PBS | #SBATCH |
| Job Name |  	-N <*name*>  | 	--job-name=<*name*> OR -J <*name*> |
| Node Count |  	-l nodes=<*count*> |  	--nodes=<*minnodes[-maxnodes]*> OR -N <*minnodes[-maxnodes]*> |
| CPU Count |  	-l ppn=<*count*> |  	--ntasks-per-node=<*count*> |
| CPUs Per Task |  	  	 | --cpus-per-task=<*count*> |
| Memory Size |  	-l mem=<*MB*> 	 | --mem=<*MB*> OR --mem-per-cpu=<*MB*> |
| Wall Clock Limit |  	-l walltime=<*hh:mm:ss*> | 	--time=<*min*> OR --time=<*days-hh:mm:ss*> |
| Standard Output File |  	-o <*file_name*> |  	--output=<*file_name*> OR -o <*file_name*> |
| Job Arrays |  	-t <*array_spec*> |  	--array=<*array_spec*> OR -a <*array_spec*> |
| Standard Error File |  	-e <*file_name*> |  	--error=<*file_name*> OR -e <*file_name*> |
| Combine stdout/stderr |  	-j oe (both to stdout) |  	(Default if you don’t specify --error) |
| Delay Job Start |  	-a <*time*> |  	--begin=<*time*> |

## Module Names

The machines may have different module names for the same compiler toolchain or different version number. Check the latest [Installed Software](../development/which_software_is_installed.md) page to see the module name and version. Below are some common module names that are
used in Fram.

* `intel` - loads the compiler and the MPI toolchain. Previously, a separate MPI or MPT module needed to be loaded.
* `ÒpenMPI` - loads the OpenMPI toolchain including the Intel compiler.

## MPI Tasks

On Fram, MPI tasks are run using Slurm's srun command. When porting to Fram, change the command that executes the application.

| Action     | Other HPC Machines     | Fram     |
| :------------- | :------------- | :------------- |
| Run MPI job |  	mpirun <*application*> or mpiexec <*application*> |  	srun <*application*> |
