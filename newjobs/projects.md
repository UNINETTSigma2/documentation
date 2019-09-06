# Projects and Accounting

All jobs are run in a _project_.  Use `--account` in job scripts to
select which project the job should run in.  (The queue system calls
projects _accounts_.)  Each project has a cpu hour quota, and when a
job runs, it cpu hours are subtracted from the project quota.  If there
is not enough hours left on the quota, the job will be left pending
with a reason `AssocGrpBillingMinutes` (`AssocGrpCPUMinutesLimit` on
Fram, but this will soon change.)

To see which projects you have access to on a cluster, run

    projects

The command `cost` gives an overview of the cpu hour quota.  It can be
run in different ways:

    cost                 # Show quota information for all projects you have access to
    cost -p YourProject  # Show quota information for project YourProject
    cost --details       # Adds information about how much each user has run

See `cost --man` for other options, and explanation of the output.
The `cost` command only shows usage in the current _allocation
period_.  Historical usage can be found in
<https://www.metacenter.no/mas/projects>.

## Accounting

The term "cpu hour" above is an over-simplification.  Jobs are
accounted for both cpu and memory usage, as well as usage of GPUs.
(Currently, jobs on Fram ar only accounted for their cpu usage, but
this will change soon.)

Accounting is done in terms of _billing units_, and the quota is in
_billing unit hours_.  Each job is assigned a number of billing units
based on the the requested cpus, memory and GPUs.  The number that is
subtracted from the quota is the number of billing units multiplied
with the (actual) wall time of the job.

The number billing units of a job is calculated like this:

1. Each requested cpu is given a cost of 1.
2. The requested memory is given a cost based on a _memory cost factor_
   (see below).
3. Each requested GPU is given a cost based on a _GPU cost factor_
   (see below).
4. The number of billing units is the _maximum_ of the cpu cost, memory
   cost and GPU cost.

Thus a job that asks for many cpus but little memory will be billed
for its number of cpus, while a job that asks for a lot of memory but
few cpus, will be billed for its memory requirement.

The _memory cost factor_ varies between the partitions on the
clusters:

- For the _bigmem_ partition on Saga, the factor is currently
  0.1059915 units per GiB.  This means that for a job requesting all
  memory on one of the "small" bigmem nodes, the memory cost is 40,
  while for a job requesting all memory on one of the large nodes,
  it is 320.

- For all other partitions, the factor is set such that the memory
  cost of requesting all the memory on a node is the same as
  requesting all cpus on the node (40 for normal nodes and 24 for the
  GPU nodes).

The _GPU cost factor_ on Saga is currently 6, which means that a job
requesting all 4 GPUs on a node, will cost the same as a job
requesting all cpus on it.  Note that this factor might increase in
the future, because GPUs are more expensive than cpus.
