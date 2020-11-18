

# Projects and accounting

- [Projects](#projects)
- [List available quota](#list-available-quota)
- [Accounting](#accounting)


### Projects

All jobs are run in a _project_.  Use `--account` in job scripts to
select which project the job should run in.  (The queue system calls
projects _accounts_.)  Each project has a CPU hour quota, and when a
job runs, CPU hours are subtracted from the project quota.  If there
is not enough hours left on the quota, the job will be left pending
with a reason `AssocGrpBillingMinutes` (`AssocGrpCPUMinutesLimit` on
Fram, but this will soon change.)

To see which projects you have access to on a cluster, run
```
$ projects
```


### List available quota

The command `cost` gives an overview of the CPU hour quota.  It can be
run in different ways:
```bash
# Show quota information for all projects you have access to
$ cost

# Show quota information for project YourProject
$ cost -p YourProject

# Adds information about how much each user has run
$ cost --details
```

See `cost --man` for other options, and explanation of the output.
The `cost` command only shows usage in the current _allocation
period_.  Historical usage can be found [here](https://www.metacenter.no/mas/projects).


(job_accounting)=

### Accounting

The term "CPU hour" above is an over-simplification.  Jobs are
accounted for both CPU and memory usage, as well as usage of GPUs.
(Currently, jobs on Fram are only accounted for their CPU usage, but
this will change soon.)

This also means that it is important that you
[do not ask for a lot more memory than you need](choosing_memory_settings.md),
otherwise you can get billed
for a lot more than you use, and your jobs may queue for a lot longer than you
would like to. In addition this can also block resources for others.

Accounting is done in terms of _billing units_, and the quota is in
_billing unit hours_.  Each job is assigned a number of billing units
based on the requested CPUs, memory and GPUs.  The number that is
subtracted from the quota is the number of billing units multiplied
with the (actual) wall time of the job.

The number billing units of a job is calculated like this:

1. Each requested CPU is given a cost of 1.
2. The requested memory is given a cost based on a _memory cost factor_
   (see below).
3. Each requested GPU is given a cost based on a _GPU cost factor_
   (see below).
4. The number of billing units is the _maximum_ of the CPU cost, memory
   cost and GPU cost.

The _memory cost factor_ varies between the partitions on the
clusters:

- For the _bigmem_ partition on Saga, the factor is currently
  0.1059915 units per GiB.  This means that for a job requesting all
  memory on one of the "small" bigmem nodes, the memory cost is 40,
  while for a job requesting all memory on one of the large nodes,
  it is 320.

- For all other partitions, the factor is set such that the memory
  cost of requesting all the memory on a node is the same as
  requesting all CPUs on the node (40 for normal nodes and 24 for the
  GPU nodes).

The _GPU cost factor_ on Saga is currently 6, which means that a job
requesting all 4 GPUs on a node, will cost the same as a job
requesting all CPUs on it.  Note that this factor might increase in
the future, because GPUs are more expensive than CPUs.
