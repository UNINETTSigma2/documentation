# NCCL on Olivia

The `NRIS/GPU` software environment provides a number of NCCL modules compiled for the system, for different CUDA and GCC versions:

```
ml avail nccl
NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1
NCCL/2.28.3-GCCcore-14.2.0-CUDA-12.8.0
NCCL/2.29.2-GCCcore-14.3.0-CUDA-12.9.1
NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0
```
When running NCCL applications that span multiple compute nodes, the off-node communication is implemented through the `aws-ofi-nccl` network plugin, which uses Slingshot and `libfabric` to transfer data. The above NCCL modules automatically load the correct plugin version.

For best performance users, who use containers should bind the libraries provided by these modules (`NCCL`, `aws-ofi-plugin`, `libfabric`) and make them available inside the containers.

## NCCL runtime configuration

At this moment (May 2026), all recent NCCL versions suffer from a data corruption issue when using GPUDirect communication in the `LL128` protocol (https://github.com/NVIDIA/nccl/issues/2001). To mitigate this problem it is crucial that on a Cray Slingshot systems the correct environment variables are used with the NCCL library. The current settings recommended by HPE (https://github.com/HewlettPackard/shs-ccl-docs/blob/main/ccl_env.sh) are automatically set when loading the NCCL modules on Olivia:

```
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_CXI_RX_MATCH_MODE=hybrid
```
Care must be taken when using a custom build of NCCL, or when using containers. If these variables are not set, data transfers can be corrupted, which is reflected in the `nccl-tests` results, e.g.
```
# nccl-tests version 2.17.9 nccl-headers=22902 nccl-library=22902
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1 maxBytes 134217728 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 206828 on   gpu-1-43 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank  1 Group  0 Pid 206829 on   gpu-1-43 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank  2 Group  0 Pid 206830 on   gpu-1-43 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank  3 Group  0 Pid 206831 on   gpu-1-43 device  3 [0039:01:00] NVIDIA GH200 120GB
#  Rank  4 Group  0 Pid 215832 on   gpu-1-47 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank  5 Group  0 Pid 215833 on   gpu-1-47 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank  6 Group  0 Pid 215834 on   gpu-1-47 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank  7 Group  0 Pid 215835 on   gpu-1-47 device  3 [0039:01:00] NVIDIA GH200 120GB
#  Rank  8 Group  0 Pid 228184 on   gpu-1-49 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank  9 Group  0 Pid 228185 on   gpu-1-49 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank 10 Group  0 Pid 228186 on   gpu-1-49 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank 11 Group  0 Pid 228187 on   gpu-1-49 device  3 [0039:01:00] NVIDIA GH200 120GB
#  Rank 12 Group  0 Pid  93770 on   gpu-1-51 device  0 [0009:01:00] NVIDIA GH200 120GB
#  Rank 13 Group  0 Pid  93771 on   gpu-1-51 device  1 [0019:01:00] NVIDIA GH200 120GB
#  Rank 14 Group  0 Pid  93772 on   gpu-1-51 device  2 [0029:01:00] NVIDIA GH200 120GB
#  Rank 15 Group  0 Pid  93773 on   gpu-1-51 device  3 [0039:01:00] NVIDIA GH200 120GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
           1             1      int8     sum      -1    24.41    0.00    0.00       0    21.26    0.00    0.00       0
           2             2      int8     sum      -1    28.02    0.00    0.00       0    21.18    0.00    0.00       0
           4             4      int8     sum      -1    20.95    0.00    0.00       0    21.05    0.00    0.00       0
           8             8      int8     sum      -1    21.01    0.00    0.00       0    22.22    0.00    0.00       0
          16            16      int8     sum      -1    21.01    0.00    0.00       0    20.80    0.00    0.00       0
          32            32      int8     sum      -1    21.37    0.00    0.00       0    55.32    0.00    0.00       0
          64            64      int8     sum      -1    29.23    0.00    0.00       0    24.23    0.00    0.00       0
         128           128      int8     sum      -1    35.74    0.00    0.01       0    35.51    0.00    0.01       0
         256           256      int8     sum      -1    36.74    0.01    0.01       0   105.19    0.00    0.00       0
         512           512      int8     sum      -1    38.00    0.01    0.03       0    37.45    0.01    0.03       0
        1024          1024      int8     sum      -1    39.38    0.03    0.05       0    38.84    0.03    0.05       0
        2048          2048      int8     sum      -1    42.59    0.05    0.09       0    42.34    0.05    0.09       0
        4096          4096      int8     sum      -1    47.04    0.09    0.16       0    46.63    0.09    0.16       0
        8192          8192      int8     sum      -1    53.29    0.15    0.29       0    52.20    0.16    0.29       0
       16384         16384      int8     sum      -1    60.40    0.27    0.51       0    53.04    0.31    0.58       0
       32768         32768      int8     sum      -1    55.10    0.59    1.12       0    54.44    0.60    1.13       0
       65536         65536      int8     sum      -1    75.68    0.87    1.62       0    63.65    1.03    1.93       0
      131072        131072      int8     sum      -1   597.83    0.22    0.41       0   452.35    0.29    0.54       0
      262144        262144      int8     sum      -1    76.61    3.42    6.42       0    76.21    3.44    6.45       0
      524288        524288      int8     sum      -1    92.47    5.67   10.63       0    92.14    5.69   10.67       0
     1048576       1048576      int8     sum      -1   130.34    8.05   15.08       0   130.86    8.01   15.02       0
     2097152       2097152      int8     sum      -1  6249.20    0.34    0.63   18134  6142.32    0.34    0.64    9609
     4194304       4194304      int8     sum      -1  9489.31    0.44    0.83   96817  10542.7    0.40    0.75  124342
     8388608       8388608      int8     sum      -1  9733.76    0.86    1.62  763876  9510.45    0.88    1.65  530181
    16777216      16777216      int8     sum      -1  4981.64    3.37    6.31  404221  5090.43    3.30    6.18  103664
    33554432      33554432      int8     sum      -1  9994.01    3.36    6.30  951005  8284.95    4.05    7.59  820604
    67108864      67108864      int8     sum      -1  34125.2    1.97    3.69  113492  43656.8    1.54    2.88       0
   134217728     134217728      int8     sum      -1  2800.93   47.92   89.85       0  3023.61   44.39   83.23       0

```

## Performance considerations
At this moment it seems that the `LL128` protocol improves performance for message sizes below ~8MB. For larger buffers it might be beneficial to turn it off using the following environment variable:
```
export NCCL_PROTO=^LL128
```
Since the previously described problem with data corruption occurs in `LL128`, disabling it is also a viable solution.