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


## Running Gaussian on Saga

No script examples are ready yet due to non-ready setup for Gaussian on Saga.

- Water input example (note the blank line at the end; `water.com`):

```{literalinclude} water.com
```
