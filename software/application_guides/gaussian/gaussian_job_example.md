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

**Please carefully inspect the job script examples whown below before submitting jobs!**

To run this example(s) create a directory, step into it, create the input file and submit the script with:

	$ sbatch fram_g16.sh


## Example for Gaussian on Fram

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
