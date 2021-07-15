# Dos and Don'ts

Here is a collection of things to do and not to do on the clusters.

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
