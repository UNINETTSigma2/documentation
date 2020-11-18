# Sample MPI Batch Script

Here is a sample batch script that demonstrates usage of various
variables and processes for a **normal** job on Saga.  (To run in
other job types, please read {ref}`job_scripts_on_saga`.

```{eval-rst}
.. literalinclude:: files/saga_mpi_job.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/saga_mpi_job.sh`
```

The actual startup of MPI application differs for different MPI
libraries.  Since this part is crucial for application performance,
please read about {ref}`running-mpi-applications`.
