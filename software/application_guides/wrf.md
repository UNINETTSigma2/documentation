---
orphan: true
---

# WRF / WPS

Weather Research and Forecasting Model (WRF) is a mesoscale numerical weather prediction system.

To find out more, visit the WRF website at: http://www.wrf-model.org

## Running WRF

| Module     | Version     |
| :------------- | :------------- |
| WRF | 3.8.0-intel-2016a-dmpar <br>3.9.1-intel-2016a-dmpar <br> |
| WPS | 3.8.0-intel-2016a-dmpar <br>3.9.1-intel-2016a-dmpar <br> |

To see available versions when logged into Fram issue command

    module spider wrf

To use WRF type

    module load WRF/<version>

specifying one of the available versions.

## License Information

WRF is in public domain. For more information, visit http://www2.mmm.ucar.edu/wrf/users/public.html

It is the user's responsibility to make sure they adhere to the license agreements.

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).


## Building from source

There are a lot of issues when building WRF from source, but most is covered in the WRF documentaion. 
Below are some hints that apply locally.

Please make sure all modules are using the same toolchain, like `intel/2021a` etc. 

When building WRF from source and using Intel compiler some small changes are needed in the `arch/configure.defaults` file. In the section for *Intel icc/ifort* the lines specifying mpiwrappers should read :
- `DM_FC = mpiifort`
- `DM_CC = mpiicc`
This is suggested in the section header. 

In addition when building on Betzy with AMD processors the `-xAVX` does not work if the main routine is built using this setting. Experience have shown that changing the  default flag on file arch/configure.defaults by modifying the line `OPTAVX = -xAVX` to `OPTAVX = -arch=core-avx2` yield the same performance.

