# Installing R packages

[R is a programming environment](https://www.r-project.org) for performing
statistical operations.  On this page we show how to install R libraries for
your projects.

```{contents} Table of Contents
```


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

Then create a directory which will hold the installed libraries:
```console
$ mkdir ${HOME}/R
```

Then start the R prompt where we do the rest:
```console
$ R
```

Use the R prompt to set the library path and install the package
(**adjust "user" to your username** in the highlighted line):
```{code-block} r
---
emphasize-lines: 2
---
# set the location for the packages to be installed
> .libPaths(c("/cluster/home/user/R", .libPaths()))

# install the package
> install.packages("somelibrary", repo="cran.uib.no")

# check whether the package can be loaded
> library(somelibrary)
```

To access the package in your scripts, you will need to add the `.libPaths` line to your scripts.


## Rscript example in a job 
In order to submit as a job, we need to include all R processing steps in an Rscript

Example Rscript


```
[user1@login-3.SAGA ~]$ cat test.rscript 

#!/usr/bin/env Rscript

.libPaths(c("/cluster/home/user1/R",.libPaths()))
library("ggplot2")
out_put = paste("Processing started on", Sys.info()["nodename"])
paste(out_put)

mean_plus_this <- function(data, addthis = 0) {
  new_data <- mean(data) + addthis
  return(new_data)
}

test_data <- c(4, 40, 9, 1)
mean_plus_this(test_data, 3)

```

Example job script to call the above Rscript

```
#!/bin/bash

#SBATCH --account=nn9999k
#SBATCH --job-name=R_script_test
#SBATCH --partition=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --time=15:00

set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module restore
module load R/4.0.0-foss-2020a

Rscript test.rscript > output.Rout 2> error.Rout

```


## License Information

R is available under several [open-source
licenses](https://www.r-project.org/Licenses). It is the user's responsibility
to make sure they adhere to the license agreements.
