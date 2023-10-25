# ORCA (quantum chemistry program)

This article concerns ORCA  general purpose tool for quantum
chemistry(https://orcaforum.kofo.mpg.de/)

The ORCA software requires a license to use it. So the recommended way is to install it yourself. As installing it yourself involves analyzing the binary to figure out which modules it is compatible with, some users might face difficulties. Until we find easier solutions for the issues, we will maintain the central installation.  

## How to install it yourself (Recommended)

* Create an account in https://orcaforum.kofo.mpg.de/ 
* Login to the site when your account is active
* Navigate to the Download section of  https://orcaforum.kofo.mpg.de/  and download the file that ends with  “Linux, x86-64, shared-version, .tar.xz Archive“. The description will be something like “Dynamically linked serial & parallel binaries linked against OpenMPI x.x”
* It is very important that you download the   Dynamically linked version as other versions will not work on our systems,even if you load the correct modules. 
* Transfer the downloaded file to the HPC system you are planning to use it on (e.g. SAGA). It is not possible to directly wget to saga at this time, you need to download it to your personal computer and then transfer it, is the only option.  
* Place it in a folder on SAGA and set the path (see example below)
* Then load the correct MPI module when you run this each time (see example)

**Here is an example for the user sabryr (please make sure to adopt it to your user account, with the correct project names):**

1. Agree to license terms and go with the version of ORCA you prefer.
2. Then download the file that uses shared libraries. For example for ORCA 5.0.4

```ORCA 5.0.4, Linux, x86-64, shared-version, .tar.xz Archive   ```

4. Transfer to saga. Example below is what I used, you need to change this to match your path

```
sabryr@jangama:~/Downloads$ rsync --progress orca_5_0_4_linux_x86-64_shared_openmpi411.tar.xz  sabryr@saga.sigma2.no:/cluster/projects/nn9999k/sabry/orca/
```

5. Login to saga (or the HPC system you are going to run ORCA) and change to the folder where the archive is and un-compress the archive. 

```tar -xf orca_5_0_4_linux_x86-64_shared_openmpi411.tar.xz```

6. Load the correct mpi module

```
module purge
module load gompi/2021a

```

7. Set the PATH and LD_LIBRARY_PATH 

```
export PATH=/cluster/projects/nn9999k/sabry/orca/orca_5_0_4_linux_x86-64_shared_openmpi411:$PATH

```
Test

```
sabryr@SAGA ORCA]$ orca --version
```

## Use the central installation

If you can not install it yourself or face issues when trying, we will maintain a central installation for the time being but access to this module is restricted. 

We have installed this software centrally but the access is restricted. This is to adhere to the licence terms of ORCA. The developers do not have a systematic procedure for obtaining the licence, so the only way at the moment is to agree to the end user licence agreement (EULA)  at the ORCA site and  download a copy of the software for yourself, to your computer (e.g. your laptop).  You do not need to install the software, just download it so you will have a chance to agree to the terms. 
 
The steps:

* Create an account in https://orcaforum.kofo.mpg.de/ 
* Login to the site when your account is active
* Navigate to the Download section of  https://orcaforum.kofo.mpg.de/  and download a copy for yourself and read and agree to terms of EULA (end user licence agreement )
* You do not need to install the software, above is to agree to the EULA terms.
* Then forward the confirmation mail from ORCA to contact@sigma2.no and request access.

We are sorry that the way to obtain a licence for ORCA is not obvious. But that is the only way at the moment.  You can get more details  from the below thread (you need an ORCA account to read this)

https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=2697

If you have further questions please ask the ORCA forum available from the above link.

