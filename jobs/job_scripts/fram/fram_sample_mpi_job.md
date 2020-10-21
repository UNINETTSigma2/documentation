# Sample MPI Batch Script

Here is a sample batch script that demonstrates usage of various
variables and processes for a **normal** job on Fram.  (To run in
other job types, please read [Fram Job Scripts](fram_job_scripts.md)).

```{eval-rst}
.. literalinclude:: files/fram_mpi_job.sh
  :language: bash
```

Download the script: <a href="files/fram_mpi_job.sh"
download>fram_mpi_job.sh</a> (you might have to right-click and select
`Save Link As...` or similar).

The actual startup of MPI application differs for different MPI
libraries.  Since this part is crucial for application performance,
please read about [how to run MPI jobs](running_mpi_jobs.md).
