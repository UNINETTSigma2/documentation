(installing-r-libraries)=

# Installing R libraries

[R is a programming environment](https://www.r-project.org) for performing
statistical operations.  On this page we show how to install R libraries for
your projects.

```{contents} Table of Contents
```


(installing-r-libraries-modules)=

## Selecting the module to load 

We recommend to use the command `module spider R` to see all available versions:

```console
$ module spider R

-----------------------------------------------------------------------------------------------------------------------------------------------------------
  R:
-----------------------------------------------------------------------------------------------------------------------------------------------------------
    Description:
      R is a free software environment for statistical computing and graphics.

     Versions:
        R/3.5.1-foss-2018b
        R/3.5.1-intel-2018b
        R/3.6.0-foss-2019a
        R/3.6.0-fosscuda-2019a
        R/3.6.0-intel-2019a
        R/3.6.1-foss-2019a
        R/3.6.2-foss-2019b
        R/3.6.2-fosscuda-2019b
        R/3.6.2-intel-2019b
        R/4.0.0-foss-2020a
        R/4.0.0-fosscuda-2020a
        R/4.0.3-foss-2020b
        R/4.0.3-fosscuda-2020b
        R/4.1.0-foss-2021a
        R/4.1.2-foss-2021b
        R/4.2.1-foss-2022a
```

Then load one of these, for instance:
```console
$ module load R/4.2.1-foss-2022a
```

We have also made bioconductor as a module with the base bioconductor packages.
Not all packages found in bioconductor repository are pre-installed. This is
because of compatibility issues, i.e. not all packages can be installed with
the same set of dependencies and to make it easier for the user to select the
exact version combinations for some packages.

```console
$ module spider bioconductor

-----------------------------------------------------------------------------------------------------------------------------------------------------------
  R-bundle-Bioconductor:
-----------------------------------------------------------------------------------------------------------------------------------------------------------
    Description:
      R is a free software environment for statistical computing and graphics.

     Versions:
        R-bundle-Bioconductor/3.8-foss-2018b-R-3.5.1
        R-bundle-Bioconductor/3.8-intel-2018b-R-3.5.1
        R-bundle-Bioconductor/3.9-foss-2019a-R-3.6.0
        R-bundle-Bioconductor/3.11-foss-2020a-R-4.0.0
        R-bundle-Bioconductor/3.12-foss-2020b-R-4.0.3
        R-bundle-Bioconductor/3.13-foss-2021a-R-4.1.0
        R-bundle-Bioconductor/3.14-foss-2021b-R-4.1.2
```


## How to install packages as a user

There might be some packages missing in the R module we have installed or maybe
you need a different version than what we have. In that case you could install
the packages yourself.

```{warning}
-  Installing packages may take a long time, especially compared with 
   installing on a Windows or Mac computer. The reason for this is that
   some packages will be compiled from source code to work on
   the operating system we have on the HPC systems. 

-  There is only restricted internet access from compute nodes, so you cannot easily install
   packages as part of your job. You need to install them using the login node
   and make sure to provide the path in the job script (examples below).
```

First log into the cluster and
load one of the R modules or R bundles, for instance:
```console
$ module restore
$ module load R/4.2.1-foss-2022a
```

Then create a directory which will hold the installed libraries. It is **very important** that you do not install libaries in your `/home` folder, as this has a strict {ref}`storage quota<storage-quota>`. You should therefore use your `/project` area (replace `nnXXXXk` with your project number). You can also share (or use others) libraries installed in the project area by using the path:
```console
$ mkdir /cluster/projects/nnXXXXk/R
```

Then start the R prompt where we do the rest:
```console
$ R
```

Use the R prompt to set the library path and install the package:
```{code-block} r
---
emphasize-lines: 2
---
# set the location for the packages to be installed
> .libPaths(c("/cluster/projects/nnXXXXk/R", .libPaths()))

# install the package
> install.packages("somelibrary", repo="cran.uib.no")

# check whether the package can be loaded
> library(somelibrary)
```

To access the package in your jobs, you will need to add the `.libPaths` line to your job-scripts.


## Keeping track of your R environment

A good way to keep track of your R environment is to use
[renv](https://rstudio.github.io/renv/articles/renv.html).  This tool makes it
possible to record and share your dependencies for better reproducibility.


## Rscript example in a job 

We have a separate page with examples for your {ref}`first-calculation`.


## Parallel R job example

```{warning}
Parallel R workers each need their own memory. Request enough memory for the
number of tasks you start.

The purpose of this example is to show how to set up a parallel R job with
Slurm. It also shows that increasing the number of cores can reduce the run
time, but that this depends on the workload and the parallel overhead.

We have tested the version below and it runs, but the scaling/speed-up is
still poor for this specific calculation.

When running jobs in parallel, please always verify that it actually scales and
that the run time goes down as you use more cores.

Often, a good alternative to run R code in parallel is to launch many
sequential R jobs at the same time, each doing its own thing.
```

Let's start with the run script (`parallel.sh`), where we ask for 20 cores:
```{code-block} bash
---
emphasize-lines: 7, 8
---
#!/bin/bash

#SBATCH --account=nn9997k
#SBATCH --job-name=example
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=20
#SBATCH --time=00:02:00

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module restore
module load R/4.2.1-foss-2022a

Rscript parallel.R > parallel.Rout
```

The `--mem-per-cpu` line is important for this example. Parallel R jobs start
multiple worker processes, so the total memory use increases with the number of
tasks.

Notice how in the R script (`parallel.R`) we indicate to use these 20 cores
and how we changed `%do%` to `%dopar%`. Instead of hard-coding the worker
count in the R code, it is better to reuse the number of tasks requested from
Slurm:
```{code-block} r
---
emphasize-lines: 23, 25, 27
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


workers <- as.integer(Sys.getenv("SLURM_NTASKS", unset = "1"))

registerDoParallel(workers)

foreach (i=1:100, .combine=c) %dopar% {
  approximate_pi()
}
```

For this particular calculation, increasing the number of workers further may
not help much because the work per iteration is small compared with the
parallel overhead. In practice, it is often worth testing a few task counts and
comparing the run time. For some workloads, many separate sequential jobs may
be a better choice than one parallel R job.


## License Information

R is available under several [open-source
licenses](https://www.r-project.org/Licenses). It is the user's responsibility
to make sure they adhere to the license agreements.
