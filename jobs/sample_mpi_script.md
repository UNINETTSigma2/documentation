# Sample MPI Batch Script

Here is a sample batch script that demonstrates usage of various variables and processes for a **normal** job (To run in other **partitions** and/or **QOSs**, please read about [different Job Types on Fram](jobtypes.md)).


###Run script example:

[include](files/slurm-MPI.sh)

###Download run script example here: <a href="files/slurm-MPI.sh" download>slurm_MPI.sh</a>

The actual startup of MPI application differs for different MPI libraries. Since
this part is crucial for application performance, please read about [how to start MPI jobs on Fram](mpi_jobs.md).
