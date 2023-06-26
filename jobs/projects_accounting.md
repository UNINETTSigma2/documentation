(projects-accounting)=

# Projects and accounting

```{contents} Table of Contents
```

## What is quota and why is it needed?

**Our compute clusters are shared and limited resources** and therefore we
divide the available compute resources in quotas and we specify compute quota
in "billing units". You can think of a billing unit as something that
corresponds to for how long one processing core could be used. For example, if
your project received 100 billing units you could use one processing core for
100 hours. You could also use 10 processing cores for 10 hours or 20 processing
cores for 5 hours or ...


## TL;DR - how to use billing units well

How billing units are computed is described below but here is what this means for you:

```{admonition} This is important
- {ref}`Do not ask for a lot more memory than you need <choosing-memory-settings>`,
  otherwise you can get billed for a lot more than you use, and your jobs may
  queue for a lot longer than you would like to. In addition this can also
  block resources for others.

- **You get billed for the resources you asked for, not what you used,
  with one exception: time**.
  Slurm cannot know how long your job will take. If you ask for 5 days but
  only use 2 hours, it will subtract "5 days worth of billing units" from your
  project/account once your job starts. 2 hours later, it will return to you
  the unused quota once your job ends. This means that if you ask for a lot
  more time than you actually need, you and your project colleagues may not be
  able to get other jobs scheduled in the meantime since Slurm will not let you
  overspend your quota.
```


## Projects

All jobs are run in a _project_ or _account_ (the Slurm queue system calls
projects _accounts_) and the account is something we always specify in our job
scripts to select which project the job should be "billed" in:
```{code-block} bash
---
emphasize-lines: 4
---
#!/bin/bash -l

# account name
#SBATCH --account=nnABCDk

# max running time in d-hh:mm:ss
# this helps the scheduler to assess priorities and tasks
#SBATCH --time=0-00:05:00

# ... rest of the job script
```

Each project has a **CPU hour quota**, and when a job runs, CPU hours are
subtracted from the project quota.  If there is not enough hours left on the
quota, the job will be left pending with a reason `AssocGrpBillingMinutes`.

To see which projects you have access to on a cluster, run (the list of your projects will differ):
```console
$ projects

nn9997k
nn9999k
nn9999o
```


## How to list available quota

The command `cost` gives an overview of the CPU hour quota.  It can be
run in different ways:

Show quota information for all projects you have access to:
```console
$ cost
```

Show quota information for a project:
```console
$ cost -p YourProject
```

Get information about how much each user has run:
```console
$ cost --details
```

See `cost --man` for other options, and explanation of the output.
The `cost` command only shows usage in the current _allocation
period_.  Historical usage can be found [here](https://www.metacenter.no/mas/projects).


## How billing units are computed

**The term "CPU hour" above is an over-simplification. Jobs are generally
accounted for both CPU and memory usage, as well as usage of GPUs.**
The accounting tries to assign a fair "price" to the amount of resources a job
requested.

Currently, jobs on Fram are only accounted for their CPU usage, but
this will change soon.

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

The _memory cost factor_ and _GPU cost factor_ vary between the partitions on the
clusters.


### Saga

- The `normal` partition: memory factor is 0.2577031 units per GiB. Thus
  the memory cost of a job asking for all memory on a node will
  be 46. This is a compromise between the two node types in the
  normal partition; they have 40 and 52 CPUs.

- For the `bigmem` partition, the factor is
  0.1104972 units per GiB.  This means that for a job requesting all
  memory on one of the "small" bigmem nodes, the memory cost is 40,
  while for a job requesting all memory on one of the large nodes,
  it is 320.

- On the `accel` partition, the memory factor is 0.06593407 units per
  GiB, and the GPU factor is 6.  This means that a job asking for all
  memory on a node, or all GPUs on a node, gets a cost of 24, the
  number of CPUs on the node.

- The `optimist` partition has the same memory factor as the `normal`
  partition.


### Betzy

- In the `normal` partition, only whole nodes are handed out, so each
  job is accounted for 128 units per node, and there is no memory
  factor.

- The `preproc` partition has a memory factor of 0.5245902 units per
  GiB, so a job asking for all memory on the node would have a cost of
  128, the number of CPUs on the node.

- The `accel` partition has a memory factor of 0.1294237 units per GiB, while
  the GPU factor is 16 units per GPU. This means that when one reserves 1 GPU
  on Betzy the billing is equivalent to reserving 16 CPU cores.


## Finding out how many billing units your job consumes

This only works for running and pending jobs. Here is an example (43 billing units):
```{code-block} console
---
emphasize-lines: 19
---
$ scontrol show job 123456

JobId=123456 JobName=example
   UserId=... GroupId=... MCS_label=N/A
   Priority=19760 Nice=0 Account=nnABCDk QOS=nnABCDk
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=5-00:08:56 TimeLimit=7-00:00:00 TimeMin=N/A
   SubmitTime=2022-10-03T13:43:14 EligibleTime=2022-10-03T13:43:14
   AccrueTime=2022-10-03T13:43:14
   StartTime=2022-10-03T13:43:14 EndTime=2022-10-10T13:43:14 Deadline=N/A
   PreemptEligibleTime=2022-10-03T13:43:14 PreemptTime=None
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2022-10-03T13:43:14 Scheduler=Main
   Partition=normal AllocNode:Sid=login-2:23457
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=c10-38
   BatchHost=c10-38
   NumNodes=1 NumCPUs=40 NumTasks=1 CPUs/Task=40 ReqB:S:C:T=0:0:*:*
   TRES=cpu=40,mem=172000M,node=1,billing=43
   Socks/Node=* NtasksPerN:B:S:C=1:0:*:* CoreSpec=*
   MinCPUsNode=40 MinMemoryNode=172000M MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/cluster/home/...
   WorkDir=/cluster/home/...
   StdErr=/cluster/home/...
   StdIn=/dev/null
   StdOut=/cluster/home/...
   Power=
```


## How do I get compute and storage quota?

The process is described here: {ref}`applying-computing-storage`.  If you are
unsure about how much to ask for and on which cluster, do not hesitate to
{ref}`contact us <support-line>`.

If you exhaust your quota and need more, the project manager can apply for
[additional quota](https://www.sigma2.no/extra-allocation).


## For how long can I use the compute quota?

Compute quota is always handed out for an allocation period. Allocation periods
run for six months (from April 1 to September 30, or October 1 to March 31). Unused
compute quota is not transferred to the next allocation period. The project
manager has to ask for new compute quota for every allocation period.

If you need a small allocation to experiment, you don't need to wait until
April or October, but can also apply in-between ({ref}`contact <support-line>`).
