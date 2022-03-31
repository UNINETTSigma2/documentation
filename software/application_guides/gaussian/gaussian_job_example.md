---
orphan: true
---

(gaussian-job-examples)=

# Gaussian job examples

To see which versions are available; type the following command after logged into the machine in question:

    module avail Gaussian

To use Gaussian, type

    module load Gaussian/<version>

specifying one of the available versions.

**Please carefully inspect the job script examples shown below before submitting jobs!**

To run this example(s) create a directory, step into it, create the input file and submit the script with:

	$ sbatch fram_g16.sh


## Running Gaussian on Fram

- Run script example (`fram_g16.sh`):

```{literalinclude} fram_g16.sh
:language: bash
```

- Water input example (note the blank line at the end; `water.com`):

```{literalinclude} water.com
```

If you see this warning in your Slurm output then this is not a reason for concern:
```text
ntsnet: WARNING: /cluster/software/Gaussian/g16_C.01/linda-exe/l302.exel may
not be a valid Tcp-Linda Linda executable Warning: Permanently added the ECDSA
host key for IP address '10.33.5.24' to the list of known hosts.
```
This is due the redirect via rscocket, which is necessary
if Gaussian is to scale satisfactory to more than 2 nodes.


## Running Gaussian on Saga

- Run script example (`saga_g16.sh`):

```{literalinclude} saga_g16.sh
:language: bash
```

- Water input example (note the blank line at the end; `water.com`):

```{literalinclude} water.com
```

## Running Gaussian on GPUs (Saga)

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

- Caffeine input example (note the blank line at the end; `caffeine.com`):

```{literalinclude} caffeine.com
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

If you find any issues with the GPU version of Gaussian, please contact us at {ref}`our support line<support-line>`.

```{note}
The timings in the table above represent a single use case, and the behavior might be
very different in other situations. Please perform simple benchmarks to check that
the program runs efficiently with your particular computational setup. Also do not
hesitate to contact us if you need guidance on GPU efficiency, see our extended
{ref}`extended-support-gpu`.
```

