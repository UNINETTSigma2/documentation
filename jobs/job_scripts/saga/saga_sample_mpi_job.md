# Sample MPI Batch Script

Here is a sample batch script that demonstrates usage of various
variables and processes for a **normal** job on Saga.  (To run in
other job types, please read [Saga Job Scripts](/jobs/job_scripts/saga_job_scripts.md)).

```{eval-rst}
.. literalinclude:: files/saga_mpi_job.sh
  :language: bash
```

Download the script: <a href="files/saga_mpi_job.sh"
download>saga_mpi_job.sh</a> (you might have to right-click and select
`Save Link As...` or similar).

The actual startup of MPI application differs for different MPI
libraries.  Since this part is crucial for application performance,
please read about [how to run MPI jobs](/jobs/guides/running_mpi_jobs.md).
