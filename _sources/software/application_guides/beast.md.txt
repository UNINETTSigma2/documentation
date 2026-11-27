# Beast

> BEAST 2 is a cross-platform program for Bayesian phylogenetic analysis of
> molecular sequences. It estimates rooted, time-measured phylogenies using
> strict or relaxed molecular clock models. It can be used as a method of
> reconstructing phylogenies but is also a framework for testing evolutionary
> hypotheses without conditioning on a single tree topology. BEAST 2 uses
> Markov chain Monte Carlo (MCMC) to average over tree space, so that each tree
> is weighted proportional to its posterior probability. BEAST 2 includes a
> graphical user-interface for setting up standard analyses and a suit of
> programs for analysing the results.

[More information can be found on Beast2's homepage.](https://www.beast2.org/)

## Running Beast

To run Beast load one of the available modules, discoverable with:

```console
$ module spider beast
```

### CPU job script

The following script can be used to run Beast on CPUs only.

```{note}
The script requests a full node's worth of CPUs on Saga, `--cpus-per-task`, you
should test for your self with your own input to see if this is necessary. The
same is also true for the `--mem-per-cpu` parameter, other input might require
more or less memory.
```

`````{tabs}
````{group-tab} Saga

```bash
#!/bin/bash

#SBATCH --account=nn<NNNN>k
#SBATCH --job-name=beast_cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=1G

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module --quiet reset
module load Beast/2.6.7-GCC-10.3.0-CUDA-11.3.1
module list

beast -beagle_cpu -threads $SLURM_CPUS_PER_TASK test.xml
```
````
`````

```{eval-rst}
:download:`test.xml <./beast/test.xml>`
```

### GPU job script

Beast supports GPU acceleration through a GPU accelerated version of
[`beagle-lib`](https://github.com/beagle-dev/beagle-lib). If your input
supports it, using GPUs should improve the performance, but always test before
submitting long running jobs as GPU jobs "cost" more than pure CPU jobs.

`````{tabs}
````{group-tab} Saga

```bash
#!/bin/bash

#SBATCH --account=nn<NNNN>k
#SBATCH --job-name=beast_cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=accel
#SBATCH --gpus=1

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module --quiet reset
module load Beast/2.6.7-GCC-10.3.0-CUDA-11.3.1
module list

beast -beagle_gpu test.xml
```
````
`````

```{eval-rst}
:download:`test.xml <./beast/test.xml>`
```

#### Performance increase

Some timing examples are listed below for the example sequences in `test.xml`,
linked above. The requested resources are chosen based on billing units, see
{ref}`projects-accounting`, where one GPU is the equivalent of six CPU cores.
Then the memory is chosen such that it will *not* be the determining factor for
the overall billing. Note that `CPU-hrs` is what your job is billed, i.e. what
you pay for the job.

| Configuration | CPUs | GPUs | MEM | Runtime | Speedup | Billing | CPU-hrs |
|---------------|------|------|-----|--------:|---------|---------|--------:|
| Reference | 1 | 0 | 4G | 13716.573s | 1.0 | 1 | 3.8 |
| 1 GPU equivalent | 6 | 0 | 6G | 4328.023s | 3.2 | 6 | 7.2 |
| Half normal node | 20 | 0 | 20G | 3998.535s | 3.4 | 20 | 22.2 |
| Full normal node | 40 | 0 | 40G | 6103.471s | 2.3 | 40 | 67.8 |
| 1 GPU | 1 | 1 | 4G | 1174.682s | 11.7 | 6 | 2.0 |
| 2 GPUs | 2 | 2 | 8G | 4153.796s | 3.3 | 12 | 13.8 |

The impression we get from the above data is that Beast is well suited for
running on GPUs, but limited to a single GPU. By using GPU acceleration for our
test example we reduce our CPU hour spending (`2.0` compared to `3.8`) while
also seeing an increase in performance (lower total runtime). In other words,
using GPUs is _both cheaper and faster_ than using the pure CPU implementation.

## License information

Beast 2 is available under the [GNU Lesser General Public License (LGPL)
version 2.1](https://github.com/CompEvol/beast2/blob/master/COPYING).

It is the **user’s** responsibility to make sure they adhere to the license
agreements.

## Citation

The homepage contains [information about how to cite
Beast](https://www.beast2.org/citation/).
