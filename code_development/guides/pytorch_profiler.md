---
orphan: true
---

(pytochprofiler)=
# Profiling GPU-accelerated Deep Learning application

We present an introduction to profiling GPU-accelerated Deep Learning (DL) models using [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Profiling is a necessary step in code development, as it permits identifying bottlenecks in an application, which in turn helps optimizing the application and thus improving performance. This introduction is limited to profiling DL-application that runs on a single-GPU. By the end of this guide, readers are expected to learn about:

- Defining the concept and the architecture of PyTorch Profiler.
- Setting up PyTorch on an HPC system using different methods:
   - Loading modules
   - Singularity container
   - Virtual environment     
- Profling a PyTorch-based application. 
- Visualising the output data on a web browser with Tensorboard plugin, in particular, the metrics:
   - GPU usage
   - Tensor cores usage
   - GPU Kernel view
   - Memory view
   - Trace view

```{contents}
:depth: 2
```

(profiler)=
## What is PyTorch Profiler
In general, the concept of profiling is based on statistical sampling, by collecting data at a regular time interval. Here, a profiler tool offers an overview of the execution time attributed to instructions of a program. In particular, it provides the execution time for each function; in addition to how many times each function has been called. Profiling analysis thus helps understanding the structure of a code, and more importantly, it helps identifying bottlenecks in an application. Examples of bottlenecks might be related to memory usage and/or identifying functions/libraries that use the majority of the computing time.

PyTorch Profiler is a profiling tool for analysing Deep Learning models, which is based on collecting performance metrics during training and inference. The profiler is built inside PyTorch API (cf. {ref}`Fig 1<fig-arch-profiler>`), and thus there is no need for installing additional libraries. It is a dynamical tool as it is based on gathering statistical data during the running procedure of a training model.

```{eval-rst}

.. _fig-arch-profiler:

.. figure:: pytorch_profiler/fig1.png
   :width: 600px
   :align: center

   Fig 1: A simplified version of the architecture of PyTorch Profiler. A complete picture of the architecture can be found [here](https://www.youtube.com/watch?v=m6ouC0XMYnc&ab_channel=PyTorch) (see the slide at 23:00 min).

```

As shown in the figure, the PyTorch API contains a python API and a C++ API. For simplicity we highlight only 

Here we list metrics collected by the profiler, which shall describe in {ref}`Section<demo>`:
- GPU usage
- Tensor cores usage (if it is enabled)
- GPU Kernel view
- Memory view
- Trace view
- Module view


Further details are provided in [slides](https://github.com/HichamAgueny/Profiling-GPU-accelerated-DL).

The code application is adapted from the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

The python application to be profiled is
"resnet18_api.py", which is specified for 4 batches and which can be adpated to a large number of batches.


## Setup Pytorch profiler in an HPC system 

Here we describe how to set up PyTorch using a singularity container.

### Setting up PyTorch container

- **Step 1**: Pull a PyTorch container image
e.g. from [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
(Note that the host system must have the CUDA driver installed and the container must have CUDA)

`$singularity pull docker://nvcr.io/nvidia/pytorch:22.12-py3`

- **Step 2**: Launch singularity container

`$singularity exec --nv -B ${MyEx} pytorch_22.12-py3.sif python ${MyEx}/resnet18_api.py`

where the container is mounted to the path `${MyEx}`, where the python application is located.

## Case example: Profling a Resnet 18 model
Here are lines of codes to enable profiling with [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
)

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(‘./out', worker_name=‘profiler'),
    record_shapes=True,
    profile_memory=True, 
    with_stack=True
) as prof:
```
To be incorporated just above the training loop
```python
#training step for each batch of input data
    for step, data in enumerate(trainloader):
          .
          .
          .
          .
       if step +1>= 10:
            break
        prof.step()
```

In the lines of the code defined above, one needs to specify some settings, which are split into three parts:
- import torch.profiler
- Specify the profiler context: i.e. which kind of activities you would like to profile .e.g. cpu, gpu or both [
- Specify the schedule: 
- Trace, record shape, profile memory, with stack

In the for loop where you are doing the training, you need to call the profile step (`prof.step()`). This will collect all the necessary input and will generate data that can be viewed with tensor board plugin.

Output of profiling will be saved in the `/out` directory 

Some advanced features:
-Option —wait:Profiling is disabled for the first step. If your training takes longer time and you don’t want to profile for the entire training loop, so you may want to wait before for some time before the profiling getting started.
-Option —warmup=N: You start collecting data after N steps for tracing.
-Option —active=M: trace to be active for M events [to not get a lot of traces].
During the active step, events will be recorded.

The most important is to track the memory allocation.

Selective profiling: first start profiling for a large training loop, and once you get an idea about where the bottleneck is, then you can select a few number of iterations for profiling.

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

## Launching a PyTorch-based application on an HPC system.  
For completeness, we provide a generic example of a job script that incorporates running PyTorch singularity container, and which can be adapted according to the selective computing resources.

```bash
#!/bin/bash -l
#SBATCH --job-name=PyTprofiler
#SBATCH --account=<project_account>
#SBATCH --time=00:10:00     #wall-time 
#SBATCH --partition=accel   #partition 
#SBATCH --nodes=1           #nbr of nodes
#SBATCH --ntasks=1          #nbr of tasks
#SBATCH --ntasks-per-nodes  #nbr of tasks per nodes (nbr of cpu-cores)
#SBATCH --cpus-per-task=1   #nbr of threads
#SBATCH --gpus=1            #total nbr of gpus
#SBATCH --gpus-per-node=1   #nbr of gpus per node
#SBATCH --mem=4G            #main memory
#SBATCH -o PyTprofiler.out

#define paths
Mydir=<Path-to-Workspace>
MyContainer=${Mydir}/Container/pytorch_22.12-py3.sif
MyExp=${Mydir}/examples

#specify bind paths by setting the environment variable
#export SINGULARITY_BIND="${MyExp},$PWD"

#TF32 is enabled by default in the NVIDIA NGC TensorFlow and PyTorch containers 
#To disable TF32 set the environment variable to 0
#export NVIDIA_TF32_OVERRIDE=0

#to run singularity container 
singularity exec --nv -B ${MyExp} ${MyContainer} python3 ${MyExp}/resnet18_profiler_api_4batch.py

echo 
echo "--Job ID:" $SLURM_JOB_ID
echo "--total nbr of gpus" $SLURM_GPUS
echo "--nbr of gpus_per_node" $SLURM_GPUS_PER_NODE
```

More details about how to write a job script can be found [here](https://documentation.sigma2.no/jobs/job_scripts.html).

# Conclusion
In conclusion, we have provided a guide on how to perform code profiling of GPU-accelerated Deep Learning models using PyTorch Profiler. The particularity of the profiler relies on its simplicity and ease to use without installing additional packages and with a few lines of codes to be added. These lines of code constitue the setting of the profiler, which can be customised according to the desired outcome of profiling. ... Collecting performance metrics, in particular, a summary of GPU usage including Tensor cores usage (if it is enabled). It offers, among other metrics a view of GPU kernel, memory peaks in time and Trace... These features are key elements for identifying bottlenecks in an application, in the aim of optimizing it to run efficiently and reliably. 


# Relevant links

[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

[NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

[Local port forwarding](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding)

[Slides](https://github.com/HichamAgueny/Profiling-GPU-accelerated-DL)

[PyTorch Profiler video](https://www.youtube.com/watch?v=m6ouC0XMYnc&ab_channel=PyTorch)



