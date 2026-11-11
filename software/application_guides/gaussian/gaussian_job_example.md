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
* {ref}`running-jobs`
* {ref}`job-scripts`

## Finding available Gaussian versions and submitting a standard Gaussian job
To see which versions of Gaussian software which are available on a given machine; type the following command after logged into the machine in question:

    module avail Gaussian

To use Gaussian, type

    module load Gaussian/<version>

specifying one of the available versions.

**Please inspect the job script examples before submitting jobs!**

To run an example - create a directory, step into it, create an input file (for example for water - see below), download one of the job scripts shown in the examples below and submit the script with:

	$ sbatch saga_g16.sh


## Gaussian input file examples

- Water input example (note the blank line at the end; `water.com`):

```{literalinclude} water.com
```

- Caffeine input example (note the blank line at the end; `caffeine.com`):

```{literalinclude} caffeine.com
```


## Running Gaussian on Saga

On Saga there are some restrictions and tricky situations to consider. First and foremost, there is a heterogenous setup with some nodes having 52 cores and most nodes having 40 cores. Secondly, on Saga there is a 256 core limit, efficiently limiting the useful maximum amount of nodes for a Gaussian job to 6. And third, since you do share the nodes by default - you need to find a way to set resource allocations in a sharing environment not necessarily heterogenous across your given nodes.

Currently, we are working to find a solution to all these challenges and as of now our advices are:
Up to and including 2 nodes should can be done with standard advices for running jobs on Saga.
For 3 nodes and above you either need to run with full nodes or using the slurm exclusive flag:  `#SBATCH--exclusive`. We prefer the latter due to robustness.  

To facilitate this, the g16 wrapper has been edited to both be backwards compatible and adjust for the more recent insight on our side. If you are not using this wrapper, please look into the wrapper to find syntax for using in your job script. Wrapper(s) are all available in Gaussian Software folder. Current name is g16.ib.

 
- Job script example (`saga_g16.sh`):

```{literalinclude} saga_g16.sh
:language: bash
```


## Running Gaussian with Linda parallelization

Running Gaussian in Betzy and Olivia is slightly different than running it on Saga, and needs some one-time setup steps. These steps are outlined in {ref}`gaussian-resource-utilization`. Following are just the example slurm sections and input files for some simple calculations.


### Running Gaussian on Betzy

- The input file (`water_Linda.com`):
```{literalinclude} water_Linda.com
:language: bash
```


```{note}
Gaussian Defaults to 1 CPU if none are specified. Changing the variable `--cpus-per-task` in the submit script is not sufficient to increase the Gaussian's CPU usage. The user must also change the line `%NPROCSHARED` in the input file with the correct number of CPUs needed. These only count the CPUs used per Linda single worker.
```

- Job script example (`betzy_g16.sh`):

```{literalinclude} betzy_g16.sh
:language: bash
```

```{note}
Since the minimum number of nodes per job is 4 in betzy, only heavy calculations should be ran such that each node's CPU utilization is maximized. Saga and Olivia are better suited for smaller calculations.
```

### Running Gaussian on Olivia

Here is an example of a script for running Gaussian on Olivia. The same input file is used.
- The input file ("water_Linda.com`):
```{literalinclude} water_Linda.com
:language: bash
```

- Job script example (`olivia_g16.sh`):

```{literalinclude} olivia_g16.sh
:language: bash
```
