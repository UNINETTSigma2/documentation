(access-pytorch)=

# PyTorch Software Options on Olivia

Olivia supports three distinct ways to run PyTorch workloads:

1. **NRIS Module** through the NRIS GPU module stack.
2. **Container Implementation** using Apptainer explicitly.
3. **EESSI Module** using the EESSI software stack.


The PyTorch overview page ({ref}`pytorch-on-olivia`) links to the scaling guides and supporting reference pages. This page focuses only on the software choices available to you.

For validated examples of all three approaches, see {ref}`pytorch-single-gpu` for single-GPU, {ref}`pytorch-multi-gpu` for multi-GPU on a single node, and {ref}`pytorch-multi-node` for multi-node runs.

## Software Options

`````{tabs}
````{group-tab} NRIS Module
The module path is the main user-facing solution on Olivia. It looks like a normal module workflow, but the runtime is container-backed underneath.

You can load the PyTorch module like this:

```bash
ml reset
ml load NRIS/GPU
ml PyTorch/2.12.0
```

In this stack:

1. **`ml reset`** starts from a clean module environment.
2. **`NRIS/GPU`** selects the ARM software stack used on Olivia GPU compute nodes.
3. **`PyTorch/2.12.0`** loads the user-facing PyTorch module. Under the hood, this module is a wrapper around the container-backed PyTorch runtime. It will load the GPU communication stack needed for multi-GPU and multi-node PyTorch jobs. On Olivia this also means the supporting communication layer used with NCCL, including components such as libfabric and AWS OFI NCCL.

If you want to see the container-based launch model directly, the PyTorch guide pages also show the equivalent direct-container examples for single-GPU, multi-GPU, and multi-node runs.

This is the recommended default when the published module already covers your workflow.

```{warning}
Do **not** plan around `pip install` based extension on NRIS GPU module stack. If packages are missing, use the container solution instead.
```
````

````{group-tab} Container Implementation
The direct container path uses Apptainer explicitly. Choose this path when you want direct control over the container image, bind mounts, overlays, and launch details.

This is the lower-level version of the module path. The module solution already wraps a container-backed runtime for you, while the direct container path lets you work with that style of launch directly.

The validated container used in the guide examples is:

```bash
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"
```

For actual job examples, including the validated single-GPU, multi-GPU, and multi-node launch patterns, see the direct-container tabs in {ref}`pytorch-single-gpu`, {ref}`pytorch-multi-gpu`, and {ref}`pytorch-multi-node`.

In particular, the multi-node guide shows the communication setup needed for inter-node PyTorch runs on Olivia.

```{warning}
If you need to add Python packages to the container, see {ref}`pytorch-overlay-images`.
```

````

````{group-tab} EESSI Module
The **{ref}`eessi` (European Environment for Scientific Software Installations)** software stack provides a simple module-based way to use PyTorch when the packages you need are already available in EESSI.

Typical loading looks like this:

```bash
module load EESSI/2025.06
module load torchvision/0.27.0-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
```

Use this path when the required packages are already present in the EESSI stack. Moreover, if you need some specific package for you and want us to include in EESSI stack, please let us know.


````
`````


If you want recommendations for where to keep overlays, models, datasets, and Hugging Face caches on Olivia, see {ref}`pytorch-models-datasets`.
