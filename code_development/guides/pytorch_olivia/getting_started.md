(pytorch-on-olivia)=

# PyTorch on Olivia

```{contents}
:depth: 2
```

This guide family shows how to run PyTorch on Olivia in three ways:

1. **Module path** through the NRIS GPU software stack.
2. **Direct container path** using Apptainer explicitly.
3. **EESSI path** using the EESSI software stack.

**The main focus is the HPC workflow: start on one GPU, then scale to multiple GPUs on one node, and then to multiple nodes.**

## Guide Structure

Use the reference pages first:

1. {ref}`PyTorch software options <access-pytorch>`
2. {ref}`Models, datasets, caches, and overlays <pytorch-models-datasets>`
3. {ref}`Adding Python packages to module and container paths <pytorch-overlay-images>`

Then follow the execution guides:

1. {ref}`Single-GPU guide <pytorch-single-gpu>`
2. {ref}`Multi-GPU guide <pytorch-multi-gpu>`
3. {ref}`Multi-node guide <pytorch-multi-node>`

```{admonition} Performance Summary
:class: tip

This 3-part guide walks you through scaling PyTorch training on Olivia's GH200 GPUs:

| Configuration | Throughput | Speedup |
|---------------|------------|---------|
| Single GPU (Part 1) | ~5,100 img/s | 1x |
| 4 GPUs on 1 node (Part 2) | ~37,000 img/s | 7x |
| 8 GPUs on 2 nodes (Part 3) | ~63,000 img/s | 12x |

The multi-GPU guides use FP16 mixed precision for improved performance.
```

```{note}
Key considerations for Olivia:

1. The login node is x86_64, while the GPU compute nodes are Aarch64.
2. Software and containers must therefore be compatible with ARM on the compute nodes.
3. Set up projects in project or work storage, not in your home directory.
```

```{toctree}
:hidden:
access_pytorch
models_and_datasets
overlay_images
PyTorchSingleGpu
PyTorchMultiGpu
PyTorchMultiNode
```
