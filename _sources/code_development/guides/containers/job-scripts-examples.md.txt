(job-script-apptainer)=
# Apptainer in job scripts
This part demonstrates how to run a container inside a cluster job script, execute commands within it, understand how it shares the host´s Linux kernel while running it own Operating System(OS), navigate directory permissions. Earlier we pull the `hello-world.sif` container on Saga and we will be using the same one for this experiment. 


## Core Concepts Covered:
1. `singularity exec <image>.sif <command>` : 
Runs a specific command inside the container and then immediately exits.

2. `Shared Kernel / Isolated OS` : The container uses the host cluster's  kernel, but runs its own isolated user workspace (e.g., Ubuntu running on top of a Rocky/CentOS host).

3. `Automatic Storage Mounting` : Your `$HOME` directory is mounted automatically inside the container by default.

## Create the Slurm Job Script
Create a file named `submit_container.sh`. This script will print the host system's operating system, and then ask the container to print its own internal operating system.

```{eval-rst}
.. tabs::

   .. group-tab:: Saga

      .. code-block:: bash

         #!/usr/bin/bash
         #SBATCH --account=<myaccount>
         #SBATCH --job-name=apptainer_test
         #SBATCH --ntasks=1                 # Run single instance of the application
         #SBATCH --mem=2G                   # Allocate 2GB of total RAM  for entire job
         #SBATCH --cpus-per-task=1          # Single CPU core for test, increase for internal multi-threading/parallel data loading
         #SBATCH --time=00:03:00
         #SBATCH --output=apptainer_%j.out

         echo "=== Host Operating System ==="
         cat /etc/os-release

         echo -e "\n=== Container Operating System ==="
         apptainer exec hello-world.sif cat /etc/os-release

         echo -e "\n=== Current Directory contents from inside the container ==="
         apptainer exec hello-world.sif ls -l

   .. group-tab:: Betzy

      .. code-block:: bash

         #!/usr/bin/bash
         #SBATCH --account=<myaccount>
         #SBATCH --job-name=apptainer_test
         #SBATCH --ntasks=1            # Run single instance of the application
         #SBATCH --mem=2G              # Allocate 2GB of total RAM  for entire job
         #SBATCH --time=00:03:00
         #SBATCH --output=apptainer_%j.out
         #SBATCH --nodes=1             # Single node job
         #SBATCH --cpus-per-task=1     # Single CPU core for test, increase for internal multi-threading/parallel data loading
         #SBATCH --partition=preproc   # for small pre/post processing tasks 

         echo "=== Host Operating System ==="
         cat /etc/os-release

         echo -e "\n=== Container Operating System ==="
         apptainer exec hello-world.sif cat /etc/os-release

         echo -e "\n=== Current Directory contents from inside the container ==="
         apptainer exec hello-world.sif ls -l

   .. group-tab:: Olivia

      .. code-block:: bash

         #!/usr/bin/bash
         #SBATCH --account=<myaccount>
         #SBATCH --job-name=apptainer_test
         #SBATCH --ntasks=1            # Run single instance of the application
         #SBATCH --mem=2G              # Allocate 2GB of total RAM  for entire job
         #SBATCH --time=00:03:00
         #SBATCH --cpus-per-task=1     # Single CPU core for test, increase for internal multi-threading/parallel data loading
         #SBATCH --output=apptainer_%j.out

         echo "=== Host Operating System ==="
         cat /etc/os-release

         echo -e "\n=== Container Operating System ==="
         apptainer exec hello-world.sif cat /etc/os-release

         echo -e "\n=== Current Directory contents from inside the container ==="
         apptainer exec hello-world.sif ls -l
```

```{note}
Notice that we have used partition as `preproc` on Betzy but not on Olivia and Saga. To learn different types of jobs on those machine, please refer to these pages.
1. Betzy : {ref}`job-types-betzy`
2. Saga : {ref}`job-types-saga`
1. Olivia : {ref}`job-types-olivia`
```

Once you submit the job using `sbatch submit_container.sh` the slurm output file will be generated as shown below.

```bash
=== Host Operating System ===
NAME="Rocky Linux"
VERSION="9.6 (Blue Onyx)"
ID="rocky"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.6"
PLATFORM_ID="platform:el9"
PRETTY_NAME="Rocky Linux 9.6 (Blue Onyx)"
ANSI_COLOR="0;32"
LOGO="fedora-logo-icon"
CPE_NAME="cpe:/o:rocky:rocky:9::baseos"
HOME_URL="https://rockylinux.org/"
VENDOR_NAME="RESF"
VENDOR_URL="https://resf.org/"
BUG_REPORT_URL="https://bugs.rockylinux.org/"
SUPPORT_END="2032-05-31"
ROCKY_SUPPORT_PRODUCT="Rocky-Linux-9"
ROCKY_SUPPORT_PRODUCT_VERSION="9.6"
REDHAT_SUPPORT_PRODUCT="Rocky Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="9.6"

=== Container Operating System ===
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 14.04.6 LTS"
VERSION_ID="14.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"

=== Current Directory contents from inside the container ===
total 61189
-rw-r--r-- 1 <username> nogroup     1006 Jul  7 10:32 apptainer_18837859.out
-rwxrwxr-x 1 <username> nogroup 62652447 Jul  7 10:25 hello-world.sif
-rw-rw-r-- 1 <username> nogroup      540 Jul  7 10:32 submit_container.sh
```

Notice how in the container we are on a Ubuntu operating system while the host 
is Rocky Linux on Saga. And you will see different version of operating system
on the host in our different machines.




