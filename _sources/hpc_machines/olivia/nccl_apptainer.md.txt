(nccl-apptainer)=

# NCCL and apptainer on Olivia

Achieving good performance with NCCL executed from inside a container requires using some host-side libraries:

 - `libfabric` compiled with Slingshot support
 - `NCCL` compiled with recent CUDA support
 - `aws-ofi-nccl` plugin, which implements Slingshot support for NCCL
 - `OpenMPI` with `libfabric` support

The following apptainer `def` file demonstrates how to build a base container that compiles `nccl-tests` - a set of benchmark programs designed to test NCCL performance:

```
Bootstrap: docker
From: ubuntu:25.04

%setup
    mkdir -p $APPTAINER_ROOTFS/lib64
    mkdir -p $APPTAINER_ROOTFS/usr/lib64
    mkdir -p $APPTAINER_ROOTFS/usr/local/cuda
    mkdir -p $APPTAINER_ROOTFS/cluster
    mkdir -p $APPTAINER_ROOTFS/opt/openmpi
    mkdir -p $APPTAINER_ROOTFS/opt/nccl
    mkdir -p $APPTAINER_ROOTFS/opt/libfabric
    mkdir -p $APPTAINER_ROOTFS/opt/aws-ofi-nccl
    mkdir -p $APPTAINER_ROOTFS/opt/nccl-tests
    
%post
    export DEBIAN_FRONTEND=noninteractive
    ln -fs /usr/share/zoneinfo/Europe/Oslo /etc/localtime
    echo "Europe/Oslo" > /etc/timezone

    apt-get update -y
    apt-get install -y wget bzip2 gcc g++ make python3

    # Ubuntu has a newer version of libreadline - pretend it's 7, which we need on Olivia.
    # Otherwise it can be provided with --bind /lib64/libreadline.so.7
    ln -s /usr/lib/aarch64-linux-gnu/libreadline.so.8 /lib64/libreadline.so.7
    
    # install vanilla NCCL
    cd /tmp
    wget https://github.com/NVIDIA/nccl/archive/refs/tags/v2.30.4-1.tar.gz
    tar xaf v2.30.4-1.tar.gz
    cd nccl-2.30.4-1
    make -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" src.build
    mv build/* /opt/nccl/
    cd /tmp
    rm -rf nccl-2.30.4-1

    # install vanilla OpenMPI
    cd /tmp
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.bz2
    tar xaf openmpi-5.0.10.tar.bz2
    cd openmpi-5.0.10
    ./configure --prefix=/opt/openmpi
    make -j install
    cd /tmp
    rm -rf openmpi-5.0.10

    # install nccl-tests
    cd /tmp
    wget https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.18.3.tar.gz
    tar xaf v2.18.3.tar.gz
    cd nccl-tests-2.18.3/
    make -j MPI=1 NCCL_HOME=/opt/nccl NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" src.build CC=/opt/openmpi/bin/mpicc CXX=/opt/openmpi/bin/mpicxx
    mv build/* /opt/nccl-tests
    cd /tmp
    rm -rf nccl-tests-2.18.3
    
%environment
    export PATH=/opt/nccl-tests:/usr/bin:$PATH
    export LD_LIBRARY_PATH=/opt/openmpi/lib:/opt/libfabric/lib/:/opt/nccl/lib:/opt/aws-ofi-nccl/lib:/usr/lib64:/lib64:/usr/local/cuda/lib/
```

Note that in the above script we install a basic version of `NCCL` and `OpenMPI` - only to be able to compile the application (in this case `nccl-tests`). Those libraries will not be actually be used during runtime. Instead, we will bind the host-side optimized libraries.

The container can be built on any Olivia GPU node:

```
ml load NRIS/GPU
ml load CUDA/13.0.0
apptainer build --nv --fakeroot --bind /cluster --bind $EBROOTCUDA:/usr/local/cuda nccl-tests.sif nccl-tests.def
```
To run the `nccl-tests` submit the following SLURM script:
```
#!/bin/bash

#SBATCH --job-name=nccl-tests
#SBATCH --partition=accel --gpus-per-node=4 --ntasks-per-node=4
#SBATCH --mem=700G
#SBATCH --time=1:00:00

module load NRIS/GPU
module load OpenMPI/5.0.10-GCC-14.3.0
module load NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0
module list

srun apptainer exec --nv --bind /cluster --bind $EBROOTCUDA:/usr/local/cuda --bind $EBROOTOPENMPI:/opt/openmpi --bind $EBROOTNCCL:/opt/nccl --bind $EBROOTLIBFABRIC:/opt/libfabric --bind $EBROOTAWSMINOFIMINNCCL:/opt/aws-ofi-nccl --bind /usr/lib64 nccl-tests.sif /opt/nccl-tests/all_reduce_perf -d int8 -b 1 -e 128M -f 2 -g 1
```
It's important to note that the correct environment variables used to configure `OpenMPI` and `NCCL` are set, since we load the corresponding `NRIS` modules. These variables are propagated into the container, hence assuring good performance and correctness (see [this article](nccl-olivia) for an in-depth explanation).

For example, to run on 2 GPU nodes:
```
sbatch --nodes=2 --partition=accel --account=... ./nccl-tests.job
```
And the job output:
```
# nccl-tests version 2.18.3 nccl-headers=23004 nccl-library=23004
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1 maxBytes 134217728 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 221423 on    gpu-1-1 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank  1 Group  0 Pid 221421 on    gpu-1-1 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank  2 Group  0 Pid 221424 on    gpu-1-1 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank  3 Group  0 Pid 221422 on    gpu-1-1 device  3 [0039:01:00] NVIDIA GH200 120GB
#  Rank  4 Group  0 Pid 201122 on    gpu-1-2 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank  5 Group  0 Pid 201121 on    gpu-1-2 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank  6 Group  0 Pid 201119 on    gpu-1-2 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank  7 Group  0 Pid 201120 on    gpu-1-2 device  3 [0039:01:00] NVIDIA GH200 120GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
           1             1      int8     sum      -1    26.95    0.00    0.00       0    22.40    0.00    0.00       0
           2             2      int8     sum      -1    97.16    0.00    0.00       0    22.58    0.00    0.00       0
           4             4      int8     sum      -1    21.81    0.00    0.00       0    21.97    0.00    0.00       0
           8             8      int8     sum      -1    22.80    0.00    0.00       0    22.35    0.00    0.00       0
          16            16      int8     sum      -1    22.02    0.00    0.00       0    21.81    0.00    0.00       0
          32            32      int8     sum      -1    22.36    0.00    0.00       0    22.24    0.00    0.00       0
          64            64      int8     sum      -1    24.82    0.00    0.00       0    23.89    0.00    0.00       0
         128           128      int8     sum      -1    28.81    0.00    0.01       0    28.88    0.00    0.01       0
         256           256      int8     sum      -1    32.95    0.01    0.01       0    29.16    0.01    0.02       0
         512           512      int8     sum      -1    28.76    0.02    0.03       0    28.54    0.02    0.03       0
        1024          1024      int8     sum      -1    32.35    0.03    0.06       0    31.42    0.03    0.06       0
        2048          2048      int8     sum      -1    31.32    0.07    0.11       0    30.50    0.07    0.12       0
        4096          4096      int8     sum      -1    32.19    0.13    0.22       0    32.62    0.13    0.22       0
        8192          8192      int8     sum      -1    36.09    0.23    0.40       0    33.83    0.24    0.42       0
       16384         16384      int8     sum      -1    36.59    0.45    0.78       0    35.31    0.46    0.81       0
       32768         32768      int8     sum      -1    40.00    0.82    1.43       0   123.30    0.27    0.47       0
       65536         65536      int8     sum      -1    61.21    1.07    1.87       0    59.30    1.11    1.93       0
      131072        131072      int8     sum      -1    67.95    1.93    3.38       0    88.03    1.49    2.61       0
      262144        262144      int8     sum      -1   196.97    1.33    2.33       0   179.73    1.46    2.55       0
      524288        524288      int8     sum      -1    93.05    5.63    9.86       0    92.01    5.70    9.97       0
     1048576       1048576      int8     sum      -1    90.58   11.58   20.26       0    89.09   11.77   20.60       0
     2097152       2097152      int8     sum      -1    98.79   21.23   37.15       0    98.49   21.29   37.26       0
     4194304       4194304      int8     sum      -1   150.75   27.82   48.69       0   150.59   27.85   48.74       0
     8388608       8388608      int8     sum      -1   340.35   24.65   43.13       0   215.52   38.92   68.12       0
    16777216      16777216      int8     sum      -1   383.60   43.74   76.54       0   647.17   25.92   45.37       0
    33554432      33554432      int8     sum      -1   837.94   40.04   70.08       0   931.77   36.01   63.02       0
    67108864      67108864      int8     sum      -1  1333.15   50.34   88.09       0  1395.72   48.08   84.14       0
   134217728     134217728      int8     sum      -1  2309.49   58.12  101.70       0  2182.20   61.51  107.63       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.8617 
#
# Collective test concluded: all_reduce_perf
#
```