(jobs-dos-donts)=
# Dos and Don'ts

Here is a collection of things to do and not to do on the clusters:

- Always use the queue system for running jobs.  The login nodes are
  only for file transfer, compilation, editing, job submission and
  short tests, etc. If you run production jobs on the login nodes, we
  will kill them and email you about it.  If you continue to do it, we
  might have to lock your account.
- Don't run interactive calculations on the login nodes; use [srun or
  salloc](interactive_jobs.md).
- Don't use `--exclusive` in job scripts.  It does not "play well"
  with `--mem-per-cpu`.
- Don't use `--hint=nomultithread` in jobs on Fram, at least not with
  Intel MPI.  If you do, the result is that all the tasks (ranks) will
  be bound to the first CPU core on each compute node.
- Use resources responsibly. The resources are shared among many users and 
  fairness is number one priority. Make sure that you have gone through the 
  [Slurm documentation](https://slurm.schedmd.com/squeue.html#lbAG), 
  {ref}`job-types`, {ref}`queue-system`,[HPC machines](/hpc_machines/hardware_overview.md), 
  and {ref}`choosing-memory-settings` to verify that you are submitting the right 
  job to the right partition to the right hardware and not wasting resources. 
  Also, make sure you have read and understood {ref}`support-line` before 
  submitting a ticket to our front-line support.

