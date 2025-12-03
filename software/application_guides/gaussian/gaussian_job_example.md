---
orphan: true
---

(gaussian-job-examples)=

# Gaussian NRIS machines Job Examples

```{note}
Here we present tested examples for various job types on the different NRIS machines.
This will be under more or less continoues development, and if you find things missing 
and/or not working as expected, do not hesitate to {ref}`contact us <support-line>`.
```

```{contents} Table of Contents
```

## Expected knowledge base

Before you run any Gaussian calculations, or any other calculations on NRIS machines for that matter, you are expected to update yourself on NRIS machinery specifics. A decent minimal curriculum is as follows:

* {ref}`hardware-overview`
* {ref}`getting-started`
* {ref}`getting-started`
* {ref}`running-jobs`
* {ref}`job-scripts`

## Finding available Gaussian versions and submitting a standard Gaussian job
To see which versions of Gaussian software which are available on a given machine; type the following command after logged into the machine in question:

    module avail Gaussian

To use Gaussian, type

    module load Gaussian/<version>

specifying one of the available versions.

**Please inspect the job script examples before submitting jobs!**

To run an example - create a directory, step into it, create an input file (for example for water - see below), download a job script (for example the fram cpu job script as shown below) and submit the script with:

	$ sbatch saga_g16.sh


## Gaussian input file examples

- Water input example (note the blank line at the end; `water.com`):

```{literalinclude} water.com
```

- Caffeine input example (note the blank line at the end; `caffeine.com`):

```{literalinclude} caffeine.com
```


## Running Gaussian on Saga

On Saga there are more restrictions and tricky situations to consider than on Fram. First and foremost, there is a heterogenous setup with some nodes having 52 cores and most nodes having 40 cores. Secondly, on Saga there is a 256 core limit, efficiently limiting the useful maximum amount of nodes for a Gaussian job on Saga to 6. And third, since you do share the nodes by default - you need to find a way to set resource allocations in a sharing environment not necessarily heterogenous across your given nodes.

Currently, we are working to find a solution to all these challenges and as of now our advices are:
Up to and including 2 nodes should can be done with standard advices for running jobs on Saga.
For 3 nodes and above you either need to run with full nodes or using the slurm exclusive flag:  `#SBATCH--exclusive`. We prefer the latter due to robustness.  

To facilitate this, the g16 wrapper has been edited to both be backwards compatible and adjust for the more recent insight on our side. If you are not using this wrapper, please look into the wrapper to find syntax for using in your job script. Wrapper(s) are all available in Gaussian Software folder. Current name is g16.ib.

 
- Job script example (`saga_g16.sh`):

```{literalinclude} saga_g16.sh
:language: bash
```



## Running Gaussian on GPUs on Saga

Both of the current `g16` versions on Saga supports GPU offloading, and we have provided
an alternative wrapper script for launching the GPU version. The only things that
need to change in the run script are the resource allocation, by adding `--gpus=N`
and `--partition=accel`, and to use the `g16.gpu` wrapper script instead of `g16.ib`.
The `g16.gpu` script is available through the standard Gaussian modules, `Gaussian/g16_B.01`
and `Gaussian/g16_C.01` (the latter will likely have better GPU performance since it is
the more recent version).

There are some important limitations for the current GPU version:

- It can only be run as single-node (up to 24 CPU cores + 4 GPU), so please specify `--nodes=1`
- The number of GPUs must be specified with the `--gpus=N` flag (not `--gpus-per-task`)
- The billing ratio between GPUs and CPUs is 6:1 on Saga, so the natural way to increment
resources is to add 6 CPUs per GPU
- Not all parts of Gaussian is able to utilize GPU resources. From the official [docs](https://gaussian.com/gpu/):
```text
GPUs are effective for larger molecules when doing DFT energies, gradients and frequencies
(for both ground and excited states), but they are not effective for small jobs. They are
also not used effectively by post-SCF calculations such as MP2 or CCSD.
```

 - Run script example (`gpu_g16.sh`)
```{literalinclude} gpu_g16.sh
:language: bash
:emphasize-lines: 5-8, 30
```


Some timing examples are listed below for a single-point energy calculation on the
Caffeine molecule using a large quadruple zeta basis set. The requested resources are
chosen based on billing units, see {ref}`projects-accounting`, where one GPU is the
equvalent of six CPU cores. Then the memory is chosen such that it will *not* be the
determining factor for the overall billing.

|   Configuration        | CPUs     | GPUs   | MEM       | Run time      | Speedup   | Billing   | CPU-hrs      |
|:----------------------:|:--------:|:------:|:---------:|:-------------:|:---------:|:---------:|:------------:|
|   Reference            | 1        | 0      |      4G   |  6h51m26s     |   1.0     |     1     |    6.9       |
|   1 GPU equivalent     | 6        | 0      |     20G   |  1h00m45s     |   6.8     |     6     |    6.1       |
|   2 GPU equivalents    | 12       | 0      |     40G   |    36m08s     |  11.4     |    12     |    7.2       |
|   3 GPU equivalents    | 18       | 0      |     60G   |    30m14s     |  13.6     |    18     |    9.1       |
|   4 GPU equivalents    | 24       | 0      |     80G   |    19m52s     |  20.7     |    24     |    7.9       |
|   Full normal node     | 40       | 0      |    140G   |    13m05s     |  31.4     |    40     |    8.7       |
|   1/4 GPU node         | 6        | 1      |     80G   |    22m41s     |  18.1     |     6     |    2.3       |
|   1/2 GPU node         | 12       | 2      |    160G   |    15m44s     |  26.2     |    12     |    3.1       |
|   3/4 GPU node         | 18       | 3      |    240G   |    12m03s     |  34.1     |    18     |    3.6       |
|   Full GPU node        | 24       | 4      |    320G   |    10m12s     |  40.3     |    24     |    4.1       |

The general impression from these numbers is that Gaussian scales quite well for this
particular calculation, and we see from the last column that the GPU version is
consistently about a factor two more efficient than the CPU version, when comparing
the actual consumed CPU-hours. This will of course depend on the conversion factor from
CPU to GPU billing, which will depend on the system configuration, but at least with
the current ratio of 6:1 on Saga it seems to pay off to use the GPU over the CPU
version (queuing time not taken into account).

If you find any issues with the GPU version of Gaussian, please contact us at {ref}`our support line <support-line>`.

```{note}
The timings in the table above represent a single use case, and the behavior might be
very different in other situations. Please perform simple benchmarks to check that
the program runs efficiently with your particular computational setup. Also do not
hesitate to contact us if you need guidance on GPU efficiency, see our extended
{ref}`extended-support-gpu`.
```


## Running Gaussian in Betzy and Olivia with Linda parallelization

Linda is Gaussian's method of process parallelization. Linda allows to run certain calculations in less time dute to parallelization, and it is essential in multi-node calculations of Gaussian, as it helps Gaussian communicate across nodes. Since the minimum numnber of nodes one can run jobs on in Betzy is 4, knowing how to use Linda is essential for proper usage.

More information on Linda and parallelization in Gaussian can be read in Gaussian's documentation [here](https://gaussian.com/running/) and for what specific calculations are sped up, you can read [here](https://gaussian.com/relnotes/). In both of those links, navigate to the Parallel sections.

In order to run Gaussian with `Linda` parallelization in Betzy and Olivia one needs to set up passwordless login to the nodes, authorize these nodes for login in job-script, and put the necessary lines in the Gaussian input file. Following is a guide in how to do this

### Setting up Passwordless login to the nodes
Before any job with Gaussian is done, do the following in your home directory.
__You only have to this once.__

1. Generate a keypair for the nodes. For this, run the following in the terminal:
```shell
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "$USER@<MACHINE>" -N ""
```
This will generate a passwordless keypair. Replace `<MACHINE>` with either `betzy`, or `olivia`.

2. Add the public key (ending with `.pub`) to authorized keys by running the following commands on the terminal:
```shell
touch ~/.ssh/authorized_keys
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
```

3. If you don't have a `~/.ssh/config`, you can create it by by running the following commands in the terminal, otherwise, skip to the next step
```shell
touch ~/.ssh/config
```

4. Open the `~/.ssh/config` file  and add the lines below to the end of it with your preferred editor.
```bash
Host *.<MACHINE>.sigma2.no
IdentityFile ~/.ssh/id_ed25519
BatchMode yes
StrictHostKeyChecking yes
```
replacing `<MACHINE>` with either `betzy`, or `olivia`.

### Running Gaussian using Linda workers on Betzy
Following is an example script for Betzy, later we will show the same for Olivia:
In this script we want to run a 4 node job, where each node will start 1 linda worker using 2 processes each.
- The input file ("water_Linda.com`):
```{literalinclude} water_Linda.com
:language: bash
```
Note the preamble contains the line `%NPROCSHARED` which specifies the number of cores for each worker.

- Job script example (`betzy_g16.sh`):

```{literalinclude} betzy_g16.sh
:language: bash
```
There are two important sections in the script, which will be explained below:

1. These lines:
```bash
for HN in $(scontrol show hostnames) 
do
    mkdir -p ~/.ssh
    touch ~/.ssh/known_hosts
    ssh-keygen -R "$HN" 2>/dev/null || true
    ssh-keyscan -H "$HN" >> ~/.ssh/known_hosts 2>/dev/null || true
    
    # These two lines are for testing that the connections are set up properly, not necessary, but help in case of issues.
    ssh -o BatchMode=yes "$HN" true && echo "SSH OK"
    ssh -o BatchMode=yes "$HN" true && echo "SSH OK" || echo "SSH failed"
done
```
These lines will allow Linda to login to the different nodes without a `yes|no` prompt appearing, if this prompt appears, Linda will crash. 

2. These lines:
```bash
NodeList=$(scontrol show hostnames | while read n ; do echo $n:1 ; done |  paste -d, -s)

{
printf '%%LINDAWORKERS=%s\n' "$NodeList"
cat input.com
} > input.tmp && mv input.tmp input.com
```

Will tell Gaussian how to distribute the `lindaWorkers` through the nodes, even if just one node is used. Because of this section `do echo $n:1` only one `LindaWorker` will be added to each node.
This adds a line like the following at the top of your input file

```bash
%LindaWorkers=[node_1]:1,[node_2]:1, ...
```
If you want more `LindaWorkers` per node, change the number after the colon to what you want. For more information about this specific part of the preamble, check the Gaussian documentation linked above.

```{note}
Since the minimum number of nodes per job is 4 in betzy, only heavy calculations should be ran such that each node's CPU utilization is maximized. Saga and Olivia are better suited for smaller calculations.
```

### Running Gaussian using Linda workers on Olivia

Here is an example of a script for running Gaussian on Olivia. The specific details are as shown above for betzy. The same input file is used.
- Job script example (`olivia_g16.sh`):

```{literalinclude} olivia_g16.sh
:language: bash
```

```{note}
Gaussian Defaults to 1 CPU if none are specified. Changing the variable `--cpus-per-task` in the submit script is not sufficient to increase the Gaussian's CPU usage. The user must also change the line `%NPROCSHARED` in the input file with the correct number of CPUs needed .
```





