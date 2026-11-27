(pytorch-on-olivia)=

# PyTorch on Olivia

```{contents}
:depth: 2
```

This guide family shows how to run PyTorch on Olivia in three ways:

1. **NRIS Module** through the NRIS GPU software stack.
2. **Container Implementation** using Apptainer explicitly.
3. **EESSI Module** using the EESSI software stack.

**Regardless of how you run PyTorch, you should always follow the best-practice HPC workflow for scaling: Start training on a single GPU, learn to scale across multiple GPUs on a single node, and finally scale across multiple nodes for optimal performance.**

## Guide Structure

Use the reference pages first:

1. {ref}`PyTorch software options <access-pytorch>`
2. {ref}`Models, datasets, caches, and overlays <pytorch-models-datasets>`
3. {ref}`Adding Python packages to  container paths <pytorch-overlay-images>`
4. {ref}`Monitoring the jobs & Debugging <pytorch-monitoring-debugging>`


```{note}
Please clone the project inside your working directory from this 
repo [PyTorch Project](https://github.com/UNINETTSigma2/nris-tutorials.git) using the command given below :
`git clone <github-repo>`. Once you clone the repo, use this command to go into the actual project directory ` cd pytorch-tutorial`
```


```{warning}
Due to limited space in your home directory, set up your project in your
**work or project area** (e.g., `/cluster/work/projects/nnXXXXk/username/pytorch_olivia/`).

```

Then follow the execution guides:

1. {ref}`Single-GPU guide <pytorch-single-gpu>`
2. {ref}`Multi-GPU guide <pytorch-multi-gpu>`
3. {ref}`Multi-node guide <pytorch-multi-node>`


```{admonition} Performance Summary
:class: tip



This 3-part guide walks you through scaling PyTorch training on Olivia's GH200 GPUs:

| Configuration | Throughput | Speedup |
|---------------|------------|---------|
| Single GPU (Part 1) | ~7367 img/s | 1x |
| 4 GPUs on 1 node (Part 2) | ~24,000 img/s | 3x |
| 8 GPUs on 2 nodes (Part 3) | ~37294 img/s | 5x |

```
Higher img/s (images per second) is better because it means the model can process more training data in less time.
Across Olivia’s GH200 GPUs, throughput increases substantially as more GPUs are added: from `~7,367 img/s` on one GPU to `~24,000 img/s` on four GPUs, and `~37,294 img/s` on eight GPUs. This delivers a nice overall speedup, although the gains are less than perfectly linear due to the communication and synchronization overhead involved in distributed training.

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
monitoring_debugging
```
