# OpenMPI on Olivia

OpenMPI is the main MPI implementation used in the `NRIS/*` module environment, for both CPU and GPU nodes. It supports efficient Slingshot 11 communication through `libfabric` and the `cxi` provider. It also supports efficient in-node GPU to GPU communication through Cuda IPC. 

Compared to Cray MPI, there are some differences and configuration quirks users should be aware of. The Slingshot fabric only needs to be initialized for multi-node runs. While this is automatically taken care of in Cray MPI, OpenMPI needs to be correctly configured. This is done automatically in `NRIS` modules, but has to be dome manually by users, who use custom builds, or containers.

In multi-node runs, Cray MPI can automatically identify ranks running within the same node and use shared memory communication instead of Slingshot. In contrast, OpenMPI relies entirely on `libfabric` to implement communication, regardless of the location of the ranks. `libfabric` on the other hand uses only one provider for communication, which by default is `cxi`. This means the entire communication by default goes through Slingshot. The alternative LinkX (`lnx`) is a *multiplexing* provider capable of choosing between `shm` (shared memory), or `cxi` (Slingshot) transport backends depending on where the ranks are. Performance benefits of using LinkX are problem-dependent, and special configuration is required - especially on the GPU nodes. For those reasons by default Olivia `NRIS/*` modules use the `cxi` provider. Additional configuration required to enable `lnx` is discussed in [the next section](lnx-provider).

In general, applications can be started either with `srun`, or with `mpirun`. However, in some cases `mpirun` will not work when running single-node applications that use the `cxi` provider. In those cases `srun` must be used.


## Single-node jobs

For single-node jobs OpenMPI should be configured to use the `ob1` internal transport module, which skips initialization of `libfabric` and uses the internal shared-memory transport implementation. On the CPU partition the configuration is

```
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,self
```
On the GPU partition the `smcuda` transport should also be specified, if GPUs are used:
```
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,smcuda,self
```
If this is not done, OpenMPI will first try to initialize the `libfabric` backend. This will fail, and OpenMPI will report the following error:
```
--------------------------------------------------------------------------
Open MPI failed an OFI Libfabric library call (fi_domain).  This is highly
unusual; your job may behave unpredictably (and/or abort) after this.

  Local host: c1-3
  Location: mtl_ofi_component.c:998
  Error: Function not implemented (38)
--------------------------------------------------------------------------
```
The application will then try to use the `ob1` transport component, and consequently continue to run. Hence, the above settings are not strictly necessary, but provide a cleaner way to run the applications.

## Multi-node jobs

For multi-node jobs OpenMPI should be configured to use the `libfabric` backend, which implements optimized Slingshot communication:
```
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
```
The above settings will force the use of `libfabric` on all ranks. Application will fail if `libfabric` for whatever reason cannot be initialized on some ranks. This is crucial to assure good performance: by default OpenMPI will silently fall back to, e.g., `tcp` transport if `libfabric` initialization fails, which will severely affect the communication performance.

### `cxi` provider

By default, in the `NRIS/*` module environment `libfabric` is configured to use the `cxi` provider:

```
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
```
In this setup Slingshot communication is used also for MPI ranks that run on the same compute node. This configuration might limit the peak communication bandwidth, but provides lower latency for small messages below ~100kB - especially for GPU to GPU communication.

By default, using `cxi` for single-node runs will fail, because Slingshot is not initialized in such cases. To force the use of `libfabric` in this case the job can be submitted with 

```
sbatch --network=single_node_vni ...
```
and the application has to be started with `srun` (i.e., not OpenMPI's `mpirun`).

(lnx-provider)=

### `lnx` provider

This provider automatically chooses between the core `shm` and `cxi` providers, depending on the location of the communicating ranks. The `shm` provider uses XPMEM for CPU-to-CPU, and Cuda IPC for GPU-to-GPU communication. To enable it, the following configuration is needed:
```
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_SHM_USE_XPMEM=1
export FI_PROVIDER=lnx
export FI_LNX_PROV_LINKS=shm+cxi
export FI_CXI_RDZV_THRESHOLD=8192
```
Using `lnx` provides great benefit for multi-node jobs, in which there is a lot on single-node communication (especially GPU-to-GPU), and where the message sizes are large enough (512kB+).

On the GPU nodes there are 4 Slingshot interfaces. By default, the above `lnx` configuration results in each rank using all interfaces in a multi-rail setup, which for most HPC workloads is inefficient. A more important problem is that this configuration results in a crash with the current versions of OpenMPI and `libfabric`. To correctly enable `lnx` on the GPU nodes one has to specifically define a single `cxi` interface for each rank. This can be done, e.g., in a shell script, which at the same time defines, which GPU should be used by each rank. Assuming 4 ranks are executed per GPU node
```
# submit the slurm script
sbatch --partition=accel --gpus-per-node=4 --ntasks-per-node=4 [...]
```
a wrapper `gpubind.sh` script can look as follows:
```
#!/bin/bash

if [ -z "$OMPI_COMM_WORLD_LOCAL_RANK" ]; then
    echo using SLURM envars
    local=${SLURM_LOCALID}
    rank=${PMIX_RANK}
else
    echo using OMPI envars
    local=${OMPI_COMM_WORLD_LOCAL_RANK}
    rank=${OMPI_COMM_WORLD_RANK}
fi

# use GPU id equal to local rank id
gpu=$((${local}))
export CUDA_VISIBLE_DEVICES=$gpu
echo rank $rank local gpu $gpu

# use the corresponding cxi interface
export FI_LNX_PROV_LINKS="shm+cxi:cxi"$gpu

# execute the command line passed in the arguments
"$@"
```
In the batch script the application should be started in the following way:
```
srun --cpu-bind=verbose,cores /path/to/gpubind.sh /path/to/application [...]
```
This assures that each GPU only uses the closest `cxi` interface directly connected to it.

The need for a wrapper script is a complication, which at this moment makes it impossible to implement `lnx` as the default transport on Olivia. However, performance benefits may be substantial and users are encouraged to test it in their jobs.

Note that the `lnx`provider is only supported for OpenMPI versions 5+.
