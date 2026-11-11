---
orphan: true
---
(gaussian-resource-utilization)=
# Memory and Core Utilization

This page contains info about special features related to the Gaussian install
made on NRIS machines.


## Gaussian over Infiniband on Saga

First note that the installed Gaussian suites are currently
Linda parallel versions, so they scale out of single nodes. In additidon, our
Gaussian installation for Saga is done with a little trick, where loading of the executable
is intercepted before launch and an alternative socket library is loaded. We have also taken care of the rsh/ssh setup in our installation procedure, to avoid `.tsnet.config` dependency on user level. This
enables Gaussian to run on Infiniband network with native IB protocol, giving us two
advantages:

* The parallel fraction of the code scales to more cores.
* The shared memory performance is significantly enhanced (small scale performance).

To run Gaussian in parallel on two or more nodes requires the
additional keywords `%LindaWorkers` and `%NProcshared` in the Link 0 part of
the input file. In addition, since we do the interception-trick, we need to add the specific IB network node address into the input file. This is taken care of by a wrapper script (`g16.ib`) around
the original binary in each individual version folder. Please use this example(s) as starting point(s) when submitting jobs.

Advised job submit syntax using the wrapper:

	g16.ib $input.com > g16_$input.out

## Parallel scaling

Gaussian is a rather large program system with a range of different binaries,
and users need to verify whether the functionality they use is parallelized and
how it scales both in terms of core and memory utilization.

### Core utilization

Due to the preload Infiniband trick, we have a somewhat more generous policy
when it comes to allocating cores/nodes to Gaussian jobs. 

- We strongly advice users to first
study the scaling of the code for a representative system. 
- **Please do not reuse scripts inherited from others without studying the
performance and scaling of your job**. We recommend to take our
{ref}`gaussian-job-examples` as a starting point.

Due to its versatility, it is hard to be general about Gaussian binary scaling. We do know that plain DFT-jobs with rather non-complicated molecules like caffeine scales easily up to the core-count limit on Saga. On the other side, jobs with transition-metal containing molecules like Fe-corroles scales moderately outside of 2 full nodes on Saga. On a general note, the range on 2-4 nodes seems to be within decent scaling behaviour for most linda-executables (the LXYZ.exel-binaries, see the Gaussian home folder on each NRIS machine).

### Memory utilization:

The `%mem` allocation of memory in the Gaussian input file means two things:

- In general, it means memory/node – for share between `nprocshared`, and
  additional to the memory allocated per process. This is also documented by
  Gaussian.
- For parallel jobs it also means the memory allocation hte LindaWorker will make, but from Linda9 and onwards the need for a memory allocation on the master node has been removed. 

However, setting %mem to a value of less than 80% of the physical memory (the actual number depends on the actual node since we have standard-, medium-memory-, and big-memory nodes, see {ref}`saga`) is good practice since other system buffers, disk I/O and others can avoid having the system swap more than necessary. This is especially true for post-SCF calculations. To top this, the `%mem` tag is also influencing on performance; too high makes the job go slower, too low makes the job
fail.

Please consider the memory size in your input if jobs fail. Our job example is
set up with 500MB (which is actually a bit on the small side), test-jobs were
ran with 2000MB. Memory demand also increases with an increasing number of
cores. But this would also limit the size of problems possible to run at a
certain number of cores. For a DFT calculation with 500-1500 basis sets, `%mem=25GB`
can be a reasonable setup. 

**Note that "heap size" is also eating from the memory
pool of the node (see below).**

## Gaussian on the other machines (Olivia, Betzy)
As mentioned above, Gaussian parallelization is done through Linda. The installation on Olivia and Betzy is different than the one in Saga, as we have to connect the linda workers through ssh into the compute nodes. 

In essence the code works the same, but several additional steps need to be taken into account when submitting jobs. The main difference is that there is no wrapper script, so all additional multinode input needs to be put into the link 0 part of the input file by the users themselves. these keywords are `%LindaWorkers` and `%NProcshared`.

Due to how Gaussian scales in single worker calculations one should start adding linda workers between the 16-32 CPU mark. Some more details on scalability of jobs is outlined in {ref}`gaussian-tuning` which is a recommended read in case of bigger Gaussian calculations.

There are generally two additional steps to take when running Gaussian on Olivia and Betzy, these are ensuring passworless and promptless login to compute nodes for Linda, and specifying the node addresses to the input file in question. The details for these are outlined below.

Because Betzy enforces a minimum allocation of 4 nodes, all Gaussian jobs on Betzy must be configured for multi-node Linda execution

More information on Linda and parallelization in Gaussian can be read in Gaussian's documentation [here](https://gaussian.com/running/) and for what specific calculations are sped up, you can read [here](https://gaussian.com/relnotes/). In both of those links, navigate to the Parallel sections.

### Setting up Linda SSH access to compute nodes (one time setup)
This is done once to allow for passwordless ssh-access for Linda to the compute nodes.

1. Generate a keypair for the nodes. For this, run the following in the terminal:
```shell
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "$USER@<MACHINE>" -N ""
```
This will generate a passwordless keypair. Replace `<MACHINE>` with either `betzy`, or `olivia`.

2. Add the public key (ending with `.pub`) to authorized keys by running the following commands on the terminal:
```shell
touch ~/.ssh/authorized_keys
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

3. If you don't have a `~/.ssh/config`, you can create it by by running the following commands in the terminal, otherwise, skip to the next step
```shell
touch ~/.ssh/config
```

4. Open the `~/.ssh/config` file  and add the lines below to the end of it with your preferred editor.
For Betzy:
```text 
Host *.betzy.sigma2.no b*
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

for Olivia:
```text
Host *.olivia.* o*
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

### Specifying compute nodes used in calculation
We now need to ensure there is no more prompts that appear when Linda attempts a ssh connection. This is done by adding the compute nodes to the list of known_hosts.
To do this, these commands need to be run before any gaussian calculation, either by shell or by slurm submit script.
```bash
for HN in $(scontrol show hostnames) 
do
    mkdir -p ~/.ssh
    touch ~/.ssh/known_hosts
    ssh-keygen -R "$HN" 2>/dev/null || true
done
```


In Saga the gaussian wrapper takes care of keywords `%LindaWorkers`, `%NProcshared` and `%mem` in the Link 0 part of
the input file. This is not the case for the Olivia and Betzy installations. The user needs to add these parts themselves, specially since they are dependent on the specific nodes Gaussian is run on. While the two keywords `%NProcshared` and `%mem` can be arranged before running, the last one, `%LindaWorkers`, is per-node specific, and must be written after a resources have been allocated.
This means that the user must ensure a line like the following is at the Link 0 part of the input file 
```bash
%LindaWorkers=[node_1]:2,[node_2]:2, ...
```
where `node_1`, `node_2` and so on are the node names used for ssh-oing into. The number after the colon describes the number of linda workers per node. 

The following lines must be added ran before a gaussian calculation, either through shell or slurm script, to make sure that the nodes are properly added in the way shown above. 
```bash
NodeList=$(scontrol show hostnames | while read n ; do echo $n:2 ; done |  paste -d, -s)

{
printf '%%LINDAWORKERS=%s\n' "$NodeList"
cat input.com
} > job_runtime.com
```

