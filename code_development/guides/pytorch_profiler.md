---
orphan: true
---

(pytochprofiler)=
# Profiling GPU-accelerated Deep Learning

We present an introduction to profiling GPU-accelerated Deep Learning (DL) models using [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Profiling is a necessary step in code development, as it permits identifying bottlenecks in an application, which in turn helps improving performance. This introduction is limited to profiling DL-application that runs on a single-GPU. By the end of this tutorial, readers are expected to learn about:

- What is PyTorch Profiler.
- How to setup PyTorch Profiler on an HPC system using different methods:
   - Loading modules.
   - Singularity container.
   - Virtual environment. 
- How to view the output data on a web browser.
- How to create a Slurm script to launch a PyTorch-based application on an HPC system.
- DEMO: Profling a Resnet 18 model

```{contents}
:depth: 2
```

(profiler)=
## What is PyTorch Profiler
In general, the concept of profiling is based on statistical sampling, by collecting data at a regular time interval. Here, a profiler tool offers an overview of the execution time attributed to instructions of a program. In particular, it provides the execution time for each function; in addition to how many times each function has been called. Profiling analysis thus helps understanding the structure of a code, and most importantly, it helps identifying bottlenecks in an application. Examples of bottlenecks might be related to memory usage and/or identifying functions/libraries that use the majority of the computing time.

PyTorch Profiler is a profiling tool for analysing Deep Learning models, which is based on collecting performance metrics during training and inference. The profiler is built inside PyTorch, and thus there is no need to install additional libraries. It is a dynamical tool as it is based on gathering statistical data during the running procedure of a training model.



Further details are provided in [slides](https://github.com/HichamAgueny/Profiling-GPU-accelerated-DL).

The code application is adapted from the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

The python application to be profiled is
"resnet18_api.py", which is specified for 4 batches and which can be adpated to a large number of batches.


## Setup Pytorch profiler in HPC system 

Here we describe how to set up PyTorch using a singularity container.

### Setting up PyTorch container

- [ ] Step 0: Pull a PyTorch container image
e.g. from [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
(Note that the host system must have the CUDA driver installed and the
container must have CUDA)

`singularity pull docker://nvcr.io/nvidia/pytorch:22.12-py3`

- [ ] Step 1: Launch singularity container
`singularity exec --nv -B ${MyEx} pytorch_22.12-py3.sif python ${MyEx}/resnet18_api.py`

Here the container is mounted to the path `${MyEx}`, where the python application is located

To run this example, we have made a bash job "job.slurm" stored in the folder "/Jobs", and which can be used to run on an HPC system.


### Visualisation on a web browser

To view the output data generated from the profiling process, one needs to install TensorBord, which can be done for instance in a virtual environment

- [ ] Step0: load a python model, create and activate Virt. Env.
- Find a python module: $module avail python
- Load a python module .e.g.: `module load python/3.9.6-GCCcore-11.2.0`
- `mkdir Myenv`
- `python –m venv Myenv`
- `source Myenv/bin/activate`

- [ ] Step1: Install TensorBoard Plugi via pip wheel packages using the following command (see also [here](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)):
- `python –m pip install torch_tb_profiler`

- [ ] Step 2: Running tensorboard uisng the command:

`tensorboard --logdir=./out --bind_all` 

will generate a local address having a specific registered or private port. Note that in HPC systems, a direct navigation
to the generated address is blocked by firewalls. Therefore, connecting on a internal network from outside can be done 
via a mechanism called [local port forwarding](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding). As stated in the [SSH documentation](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding) “Local forwarding is used to forward a port from the client machine to the server machine”.

The syntax for local forwarding, which is configured using the option `–L`, can be written as, e.g.:

`ssh -L 6009:local.host:6006 username@server.address.com`

This syntax enables opening a connection to the jump server `username@server.address.com`, and forwarding 
any connection to port 6009 on the local machine to port 6006 on the server `username@server.address.com`. 

Last the local address `http://localhost:6009/` can be view in a chrome of firefox browser.




