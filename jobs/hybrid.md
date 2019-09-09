# Hybrid MPI-OpenMP Batch Job Example

Here is a sample batch script that demonstrates usage of various variables and processes for a **normal** job.

[include](files/slurm-MPI-OMP.sh)

The actual startup of MPI application differs for different MPI libraries. Since
this part is crucial for application performance, please read about [how to start MPI jobs on Fram](mpi_jobs.md).

The key to alter the number of MPI ranks per node is the kombination`--ntasks-per-node` and `--nodes`:

	#SBATCH --nodes=10 --ntasks-per-node=4 --cpus-per-task=8

The example above gives a total of 40 mpi tasks, while the jobscript example gives 20. The optimal number will vary from code to code, and is depending on implementation, type of problem and skill of programmer(s). A detailed set of advice regarding this is way outside the scope of ordinary user documentation - please search elsewhere for more info.