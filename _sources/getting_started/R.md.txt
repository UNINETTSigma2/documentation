(first-r-calculation)=

# First R calculation

Our goal on this page is to get an R calculation to run
on a compute node, both as serial and parallel calculation.

```{contents} Table of Contents
```


## Simple example to get started

We will start with a very simple R script (`simple.R`):
```r
print("hello from the R script!")
```

We can launch it on {ref}`saga` with the following job script (`simple.sh`).
**Before submitting**, adjust at least the line with `--account` to match your
allocation:
```{code-block} bash
---
emphasize-lines: 3
---
#!/bin/bash

#SBATCH --account=nn9997k
#SBATCH --job-name=example
#SBATCH --partition=normal
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module restore
module load R/4.2.1-foss-2022a

Rscript simple.R > simple.Rout
```

Submit the example job script with:
```console
$ sbatch simple.sh
```


## Longer example

Here is a longer example that takes ca. 25 seconds (`sequential.R`):
```r
library(foreach)


# this function approximates pi by throwing random points into a square
# it is used here to demonstrate a function that takes a bit of time
approximate_pi <- function() {
  # number of points to use
  n <- 2000000

  # generate n random points in the square
  x <- runif(n, -1.0, 1.0)
  y <- runif(n, -1.0, 1.0)

  # count the number of points that are inside the circle
  n_in <- sum(x^2 + y^2 < 1.0)

  4 * n_in / n
}


foreach (i=1:100, .combine=c) %do% {
  approximate_pi()
}
```

And the corresponding run script (`sequential.sh`).
**Before submitting**, adjust at least the line with `--account` to match your
allocation:
```{code-block} bash
---
emphasize-lines: 3
---
#!/bin/bash

#SBATCH --account=nn9997k
#SBATCH --job-name=example
#SBATCH --partition=normal
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module restore
module load R/4.2.1-foss-2022a

Rscript sequential.R > sequential.Rout
```


## Parallel job script example

```{warning}
We have tested this example and it works but the scaling/speed-up is pretty
poor and not worth it in this example. If you know the reason, can you please
suggest a change?

When running jobs in parallel, please always verify that it actually scales and
that the run time goes down as you use more cores.

When testing this example on the desktop, the speed-up was much better.

Often, a good alternative to run R code in parallel is to launch many
sequential R jobs at the same time, each doing its own thing.
```

Let's start with the run script (`parallel.sh`), where we ask for 20 cores:
```{code-block} bash
---
emphasize-lines: 7
---
#!/bin/bash

#SBATCH --account=nn9997k
#SBATCH --job-name=example
#SBATCH --partition=normal
#SBATCH --mem=2G
#SBATCH --ntasks=20
#SBATCH --time=00:02:00

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module restore
module load R/4.2.1-foss-2022a

Rscript parallel.R > parallel.Rout
```

Notice how in the R script (`parallel.R`) we indicate to use these 20 cores
and how we changed `%do%` to `%dopar%`:
```{code-block} r
---
emphasize-lines: 23, 25
---
library(parallel)
library(foreach)
library(doParallel)


# this function approximates pi by throwing random points into a square
# it is used here to demonstrate a function that takes a bit of time
approximate_pi <- function() {
  # number of points to use
  n <- 2000000

  # generate n random points in the square
  x <- runif(n, -1.0, 1.0)
  y <- runif(n, -1.0, 1.0)

  # count the number of points that are inside the circle
  n_in <- sum(x^2 + y^2 < 1.0)

  4 * n_in / n
}


registerDoParallel(20)

foreach (i=1:100, .combine=c) %dopar% {
  approximate_pi()
}
```


## Which of the many R modules to load?

The short answer is to get an overview about available modules first:
```console
$ module spider R
$ module spider bioconductor
```

We have more information here: {ref}`installing-r-libraries-modules`


## Installing R libraries

We have a separate page about {ref}`installing-r-libraries`.
