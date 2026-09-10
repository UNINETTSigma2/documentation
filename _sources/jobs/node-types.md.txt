(node-types)=

# NRIS systems overview

NRIS systems can be categorized by the type of service they provide users. 
Generally these are: `login`, `compute`(cpu/accel), and `service`` nodes.
Below is a brief description of each type.

## Login nodes

The login nodes are where users spend most time creating, editing, (small-scale) compiling, 
and testing of their code before starting jobs. This is where users end up when ssh-ing to 
a cluster (e.g. `ssh user@olivia.sigma2.no` will lead users to one of the login nodes, e.g. 
`uan02.oligia.sigma2.no`).

All clusters have multiple (but varying amount of) login-nodes, shown by the hostname upon 
logging in (e.g. user@`login-X.saga.sigma2.no` and `uan0X@olivia.sigma2.no` - where X is node-number).

### Intended usage

The login-nodes are **shared resource**, in which multiple users are logged on and performing 
tasks simultaneously. It is therefore very important that users do not perform computation 
tasks (such as compiling bigger code, performing heavy I/O processes, running multiple 
processes occupying too much resources, etc.), as such usage by *one* user will affect 
**all users** that are currently logged in and working on the node.

All of the mentioned tasks above (and similar tasks requiring much resources) must be run as 
jobs, which will reserve either a whole node(s) or parts of that node for a specific job. In 
cases where user I/O is required ("build-and-run" situations), users can reserve {ref}`interactive sessions <interactive-jobs>`, 
where they can ssh into the provided node and do their interactive editing/testing.

Sometimes users run certain tasks that unintentionally might use too much resources, causing 
others users issues such as slow I/O reaction (or even unable to login). Most of these situations 
are short-term and unnoticable, but:

- If you are a user noticing others misusing the login nodes, please {ref}`contact support <support-line>`, 
  and we will follow up with the user(s).
- If you are the user causing this, we will do one of the following:
  1. *Notification before cancelation*: If possible, we will contact users to cancle the processes 
  themselves (e.g. create checkpoint to resume later). If the user doesn't respond shortly, we will 
  cancle all active processes.
  2. *Cancelation before Notification*: If the node is nearing unusable for others, we will 
  **immediately end all processes** associated to that user. After this we will contact the user.

Note that repetitive occurrences of login-node misuse might result in revoking of access to NRIS-services, 
so remember to {ref}`use the shared resources responsibly <responsible-use>`.

## Service nodes (Olivia)

NIRD project areas (`/nird/dp/NS...` and `/nird/dl/NS...`) are mounted on Betzy and Saga, and users can 
access these directly (**only**) from the login-nodes. But for Olivia we have 5 Service nodes (`svc1-5`). 
These nodes are intended for moving files between NIRD and the Clusters (*any process that involves work 
with larger data*, e.g. moving datasets to Olivia before running job, and moving output files back to NIRD 
after). For an automated pre/post job files on Olivia, please see {ref}`staging in/out <stage-in-stage-out>` page.

To access the service nodes on Olivia, from the login node run `ssh svc0X` (where X is between 1 - 5). 

## Compute nodes

The compute nodes are where HPC computation is performed through the `sbatch` command. These nodes are where 
users **must** run their jobs (either through `job scripts`, or interactive-sessions). This is a queued system, 
so as long as users have enough cpu/gpu hours left on their allocation period (and comply with possible 
{ref}`job types <job-types>` on relevant cluster) they can run their computations. For more information (and tips) 
please see [Running jobs](https://documentation.sigma2.no/jobs/overview.html).
