---
orphan: true
---

# R
R is a programming environment for performing statistical operations.

To find out more, visit the R website at: https://www.r-project.org

## Running R

| Module     | Version     |
| :------------- | :------------- |
| R | 4.1.2-foss-2021b <br> 4.2.1-foss-2022a <br> 4.2.2-foss-2022b <br> 4.3.2-gfbf-2023a <br> 4.4.1-gfbf-2023b <br> 4.4.2-gfbf-2024a| 



To see available versions when logged into the cluster, issue command

    module spider R

Then, to use the desired R-version, issue command

    module load R/<version>

## How to install packages
There might be some packages missing in the R modules that are installed, or maybe you need a different
version than what currently exists. In that case you could install the packages yourself. For example,
following is the procedure to install the package called XYZ by the user *user1* on  SAGA (the steps are also similar on `fram` and `betzy`):

 -  Login to saga
 -  Load the module
 -  Create a directory in which the packages should be installed.
    - **NB**: It is very important that you **do not** install the packages in your `/home` folder, as you might risk running out of [storage quota](https://documentation.sigma2.no/files_storage/quota.html#storage-quota). 
-  Run `R`

```
        [user1@login-1.SAGA ~]$ module restore
        [user1@login-1.SAGA ~]$ module load R/4.0.0-foss-2020a
        [user1@login-1.SAGA ~]$ mkdir /cluster/projects/nnXXXXk/R
        [user1@login-1.SAGA ~]$ R
```

 -  Use the R prompt to install the package
 -  Here the packages are installed in the project area. This is a preffered choice because:
    1. You will not risk exceeding your quota 
    2. All project members can link to this path in their `R` instances, instead of having to install one package multiple times.

```
      #Set the location for the packages to be installed
      > .libPaths("/cluster/projects/nnXXXXk/R")
      #Install the package
      >  install.packages("XYX", repo="cran.uib.no")
      #Check if the package can be  loaded
      > library(XYZ)
```

 - **How to use an installed package**:
   After installing, everytime the packages needed to be accessed the `.libPaths("/cluster/projects/nnXXXXk/R")` setting should be done.
   When submitting R Script as a job, the `.libPath("/cluster/projects/nnXXXXk/R")` should be
   specified before calling the package.


## License Information

R is available under several open-source licenses. For more information, visit https://www.r-project.org/Licenses/

It is the user's responsibility to make sure they adhere to the license agreements.


