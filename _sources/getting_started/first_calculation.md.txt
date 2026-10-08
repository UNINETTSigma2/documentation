(first-calculation)=
(first-r-calculation)=

# First calculation

Our goal on this page is to get a serial calculation to run on a compute node.

```{note}
For each example, the script and its Slurm job file must be in the same
working directory before you submit.
```

```{contents} Table of Contents
```


## Simple example to get started

If you are new to Slurm, start with this minimal Bash sanity check.
It confirms that submission and output work before adding R, Python, C, or Fortran.

Simple Bash script (`simple_bash.sh`):

```bash
echo "hello from the bash script!"
```

Slurm script (`simple_bash_job.sh`) to run it on {ref}`saga`:

```bash
#!/bin/bash

#SBATCH --account=<your-account>
#SBATCH --job-name=simple-bash
#SBATCH --partition=normal
#SBATCH --time=00:01:00

set -o errexit
set -o nounset

bash simple_bash.sh
```

Submit with:

```console
$ sbatch simple_bash_job.sh
```

Now we continue with simple calculation examples in R, Python, C, and Fortran.
For C and Fortran, we compile before execution. For simple examples, this can
be done directly in the Slurm script.

`````{tabs}
````{group-tab} R

```{eval-rst}
.. literalinclude:: files/simple.R
  :language: r
```
````
````{group-tab} Python

```{eval-rst}
.. literalinclude:: files/simple.py
  :language: python
```
````

````{group-tab} C

```{eval-rst}
.. literalinclude:: files/simple.c
  :language: c
```
````

````{group-tab} Fortran

```{eval-rst}
.. literalinclude:: files/simple.f90
  :language: fortran
```
````

`````

We can launch the R, Python, C, and Fortran examples on {ref}`saga` with the
following job scripts.
**Before submitting**, adjust at least the line with `--account` to match your
allocation:

`````{tabs}
````{group-tab} R
```{eval-rst}
.. literalinclude:: files/simple.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} Python
```{eval-rst}
.. literalinclude:: files/simple_python.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} C
```{eval-rst}
.. literalinclude:: files/simple_c.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} Fortran
```{eval-rst}
.. literalinclude:: files/simple_fortran.sh
  :language: bash
  :emphasize-lines: 3
```
````

`````

Submit the example job scripts with:
```console
$ sbatch <your_slurm_script>
```


## Longer example

Here is a longer example that approximates pi using a Monte Carlo method.
It runs 100 iterations, each throwing 2 million random points. This takes
roughly 1 minute in R; C and Fortran will typically be faster.

The Python version uses NumPy, which runs array operations as compiled C code
internally. This makes it faster than R here, and much faster than a plain
Python loop would be. A pure Python implementation of the same calculation
would be an order of magnitude slower than C or Fortran.


`````{tabs}
````{group-tab} R
```{eval-rst}
.. literalinclude:: files/sequential.R
  :language: R
```
````

````{group-tab} Python
```{eval-rst}
.. literalinclude:: files/sequential.py
  :language: python
```
````

````{group-tab} C
```{eval-rst}
.. literalinclude:: files/sequential.c
  :language: c
```
````

````{group-tab} Fortran
```{eval-rst}
.. literalinclude:: files/sequential.f90
  :language: fortran
```
````

`````

And the corresponding Slurm scripts.
**Before submitting**, adjust at least the line with `--account` to match your
allocation:

`````{tabs}
````{group-tab} R

```{eval-rst}
.. literalinclude:: files/sequential.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} Python

```{eval-rst}
.. literalinclude:: files/sequential_python.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} C

```{eval-rst}
.. literalinclude:: files/sequential_c.sh
  :language: bash
  :emphasize-lines: 3
```
````

````{group-tab} Fortran

```{eval-rst}
.. literalinclude:: files/sequential_fortran.sh
  :language: bash
  :emphasize-lines: 3
```
````

`````

## Next steps

**R**
- To find available R modules: `module spider R` or `module spider bioconductor`
- For selecting the right module: {ref}`installing-r-libraries-modules`
- For installing additional R packages and running parallel R jobs: {ref}`installing-r-libraries`

**Python**
- Use `module spider Python` to see available Python modules
- Consider using containers to manage Python environments: {ref}`dev-guides_containers`

**C and Fortran**
- For compiler options and optimization flags, see {doc}`/code_development/compilers`
- For building more complex projects: {doc}`/code_development/building`
