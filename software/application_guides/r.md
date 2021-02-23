# R
R is a programming environment for performing statistical operations.

To find out more, visit the R website at: https://www.r-project.org

## Running R

| Module     | Version     |
| :------------- | :------------- |
| R |3.4.0-intel-2017a-X11-20170314 <br> 3.4.3-intel-2017b-X11-20171023 <br> 3.4.4-intel-2018a-X11-20180132 <br>|

To see available versions when logged into Fram issue command

    module spider R

To use R type

    module load R/<version>

## How to install packages
There might be some packages missing in the R module we have installed or may be you need a different
version than what we have. In that case you could install the packages yourself. For example,
following is the procedure to install the package called XYZ by the user *user1* on  SAGA. 
Please Please remmeber to use your username instead of *user1*

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
      > .libPaths("/cluster/home/user1/R")
      #install the package
      >  install.packages("XYX", repo="cran.uib.no")
      #Check if the package can be  loaded
      > library(XYZ)
```

 - How to use an installed package
   After installing, everytime the packages needed to be accessed
   The `.libPaths("/cluster/home/user1/R")` setting should be done.
   When submitting R Script as a job, the `.libPath("/cluster/home/user1/R")` should be
   specified before calling the package.


## License Information

R is available under several open-source licenses. For more information, visit https://www.r-project.org/Licenses/

It is the user's responsibility to make sure they adhere to the license agreements.


