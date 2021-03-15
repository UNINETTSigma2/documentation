# R 
R is a programming environment for performing statistical operations.

To find out more, visit the R website at: https://www.r-project.org

## Selecting the correct module load 
| Module     | Version     |
| :------------- | :------------- |
| R | R/3.5.1-foss-2018b <br> R/3.5.1-intel-2018b <br> R/3.6.0-foss-2019a <br> 
      R/3.6.0-fosscuda-2019a <br> R/3.6.0-intel-2019a <br> R/3.6.1-foss-2019a <br> 
      R/3.6.2-foss-2019b  <br> R/3.6.2-fosscuda-2019b <br> R/3.6.2-intel-2019b <br> 
      R/4.0.0-foss-2020a <br> R/4.0.3-foss-2020b <br> |

To see available versions when logged into Fram issue command

    module spider R


To use R type

    module load R/<version>

```{note}
If you do not have specific requirements please use the latest,
as of now it is  "R/4.0.0-foss-2020a"
```

## How to install packages as a user
There might be some packages missing in the R module we have installed or may be you need a different
version than what we have. In that case you could install the packages yourself. For example,
following is the procedure to install the package called XYZ by the user *user1* on  SAGA. 
Please remmeber to use your username instead of *user1*

```{note}
-  Installing packages may take a long time, specailly if your comparing with 
   installing on a Windows or Mac computer. The reason for this is that unlike
   Windows or Mac some packages will be compiled from source code to work on
   the operating system we have on the HPC systesms. 

-  There is no internet access from compute nodes, so you can not install
   packages as part of your job. You need to install them using the login node
   and make sure to provide the path in the job script (examples below)
```{note}

*The examples is for SAGA*
 -  Login to saga
 -  Load the module

```
        [user1@login-1.SAGA ~]$ module restore
        [user1@login-1.SAGA ~]$ module load R/4.0.0-foss-2020a
        [user1@login-1.SAGA ~]$ mkdir /cluster/home/user1/R
        [user1@login-1.SAGA ~]$ R
```

 - Use the R prompt to install the package

```
      #Set the location for the packages to be installed
      > .libPaths(c("/cluster/home/user1/R",.libPaths()))
      #install the package
      >  install.packages("XYX", repo="cran.uib.no")
      #Check if the package can be  loaded
      > library(XYZ)
```

```{note}
-  How to use an installed package
   After installing, everytime the packages needed to be accessed
   The `.libPaths("/cluster/home/user1/R")` setting should be done.
   When submitting R Script as a job, the `.libPath("/cluster/home/user1/R")` should be
   specified before calling the package.
```

## Bioconductor
We have also made bioconductor as a module with the base bioconductor packages.
Not all packages found in biocondctor repository are pre-installed. This is becasue 
compatobility issues, i.e. not all packages can be installed with the same set of 
dependancies and to make it easier for the user to select the exact version 
combinations for some packages.

## Selecting the correct module load 
| Module     | Version     |
| :------------- | :------------- |
| R |   R-bundle-Bioconductor/3.8-foss-2018b-R-3.5.1 <br>
        R-bundle-Bioconductor/3.8-intel-2018b-R-3.5.1 <br>
        R-bundle-Bioconductor/3.9-foss-2019a-R-3.6.0  <br>
        R-bundle-Bioconductor/3.11-foss-2020a-R-4.0.0 <br>
        R-bundle-Bioconductor/3.12-foss-2020b-R-4.0.3 <br>|

To see available versions when logged into Fram issue command

    module spider bioconductor

If you do not have specific requirements please use the latest, 
as of now it is  "R-bundle-Bioconductor/3.12-foss-2020b-R-4.0.3"

When you load the Bioconductor module a compatible R module and 
all supporting module that are needed to build libraries will be
loaded. 

Here is an example showing how to install a bioconductor package

```
        [user1@login-1.SAGA ~]$ module restore
        [user1@login-1.SAGA ~]$ module load R-bundle-Bioconductor/3.12-foss-2020b-R-4.0.3
        [user1@login-1.SAGA ~]$ mkdir /cluster/home/user1/R
        [user1@login-1.SAGA ~]$ R
```

 - Use the R prompt to install the package

```
      #Set the location for the packages to be installed
      > .libPaths(c("/cluster/home/user1/R",.libPaths()))
      #install the package
      >  BiocManager::install("XYZ", repo="cran.uib.no")
      #Check if the package can be  loaded
      > library(XYZ)
```

## License Information

R is available under several open-source licenses. For more information, visit https://www.r-project.org/Licenses/

It is the user's responsibility to make sure they adhere to the license agreements.


