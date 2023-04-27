---
orphan: true
---

(pytochprofiler)=
# Profiling GPU-accelerated Deep Learning

We present an introduction to profiling GPU-accelerated Deep Learning (DL) models using [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Profiling is a necessary step in code development, as it permits identifying bottlenecks in an application. This in turn helps optimize applications, thus improving performance. 

This introduction is limited to profiling DL-application that runs on a single GPU. By the end of this guide, readers are expected to learn about:

- Defining the concept and the architecture of PyTorch Profiler.
- Setting up PyTorch profiler on an HPC system. 
- Profiling a PyTorch-based application. 
- Visualizing the output data on a web browser with the Tensorboard plugin, in particular, the metrics:
   - GPU usage
   - GPU Kernel view
   - Memory view
   - Trace view
   - Module view 

```{contents} Table of Contents
```

(profiler)=
## What is PyTorch Profiler
In general, the concept of profiling is based on statistical sampling, by collecting data at a regular time interval. Here, a profiler tool offers an overview of the execution time attributed to the instructions of a program. In particular, it provides the execution time for each function; in addition to how many times each function has been called. Profiling analysis thus helps to understand the structure of a code, and more importantly, it helps to identify bottlenecks in an application. Examples of bottlenecks might be related to memory usage and/or identifying functions/libraries that use the majority of the computing time.

PyTorch Profiler is a profiling tool for analyzing Deep Learning models, which is based on collecting performance metrics during training and inference. The profiler is built inside the PyTorch API (cf. {ref}`Fig 1<fig-arch-profiler>`), and thus there is no need for installing additional packages. It is a dynamic tool as it is based on gathering statistical data during the running procedure of a training model.

```{eval-rst}

.. _fig-arch-profiler:

.. figure:: pytorch_profiler/Figs/fig00.png
   :width: 600px
   :align: center

   Fig 1: A simplified version of the architecture of PyTorch Profiler. A complete picture of the architecture can be found [here](#https://www.youtube.com/watch?v=m6ouC0XMYnc&ab_channel=PyTorch) (see the slide at 23:00 min).

```

As shown in the figure, the PyTorch API contains a Python API and a C++ API. For simplicity we highlight only the necessary components for understanding the functionality of PyTorch profiler, which integrates the following: (i) aTen operators, which are libraries of tensor operators for PyTorch and are GPU-accelerated with CUDA; (ii) Kineto library designed specifically for profiling and tracing PyTorch models; and (iii) LibCUPTI (CUDA Profiling Tool Interface), which is a library that provides an interface for profiling and tracing CUDA-based application (low-level profiling). The last two libraries provide an interface for collecting and analyzing the performance data at the level of GPU. 

Here we list the performance metrics provided by the profiler, which we shall describe in {ref}`Section<performance-metrics>`:
- GPU usage
- Tensor cores usage (if it is enabled)
- GPU Kernel view
- Memory view
- Trace view
- Module view

Further details are provided in these [slides](https://github.com/HichamAgueny/Profiling-GPU-accelerated-DL).

(setup-pytorch-profiler-in-an-hpc-system)=
## Setup Pytorch profiler in an HPC system
In this section, we describe how to set up PyTorch using a singularity container.

- **Step 1**: Pull and convert a docker image to a singularity image format:
e.g. from the [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

Note that when pulling docker containers using singularity, the conversion can be quite heavy and the singularity cache directory in `$HOME` space becomes full of temporary files. To speed up the conversion and avoid leaving behind temporary files, one can run these lines:

```console
$ mkdir -p /tmp/$USER 
$ export SINGULARITY_TMPDIR=/tmp/$USER 
$ export SINGULARITY_CACHEDIR=/tmp/$USER
```
and then

```console
$singularity pull docker://nvcr.io/nvidia/pytorch:22.12-py3
```

- **Step 2**: Launch the singularity container

```console
$singularity exec --nv -B ${MyEx} pytorch_22.12-py3.sif python ${MyEx}/resnet18_api.py
```

Here the container is mounted to the path `${MyEx}`, where the Python application is located. An example of a Slurm script that launches a singularity container is provided in the {ref}`Section<launching-a-pytorch-based-application>`

(case-example-profiling-a-resnet-18-model)=
## Case example: Profiling a Resnet 18 model
We list below the lines of code required to enable profiling with [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
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

Here is a code example of the Resnet18 model, in which profiling is enabled. The code is adapted from the [PyTorch tutorial](#https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

```{eval-rst}
.. literalinclude:: pytorch_profiler/resnet18_with_profiler_api.py
   :language: python
   :lines: 1-63
   :emphasize-lines: 34-59
   :linenos:

```

For reference, we provide here the same application but without enabling profiling. The code is adapted from the [PyTorch tutorial](#https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

```{eval-rst}
.. literalinclude:: pytorch_profiler/resnet18_without_profiler_api.py
   :language: python
   :lines: 1-44
   :emphasize-lines: 34-40
   :linenos:

```

In these lines of code defined above, one needs to specify the [setting for profiling](#https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). The latter can be split into three main parts:
- Import `torch.profiler`
- Specify the profiler context: i.e. which kind of **activities** one can profile. e.g. CPU activities (i.e. `torch.profiler.ProfilerActivity.CPU`), GPU activities (i.e. `torch.profiler.ProfilerActivity.CUDA`) or both activities.
- Specify the **schedule**; in particular, the following options can be specified:
  —*wait=l*: Profiling is disabled for the first `l` step. This is relevant if the training takes a longer time, and that profiling the entire training loop is not desired. Here, one can wait for `l` steps before the profiling gets started.

  —*warmup=N*: The profiler collects data after N steps for tracing.

  —*active=M*: Events will be recorded for tracing only during the active steps. This is useful to avoid tracing a lot of events, which might cause issues with loading the data. 

- Additional options: Trace, record shape, profile memory, with stack, could be enabled.

Note that, in the `for loop` (i.e. *the training loop*), one needs to call the profile step (`prof.step()`), in order to collect all the necessary inputs, which in turn will generate data that can be viewed with the Tensorboard plugin. In the end, the output of profiling will be saved in the `/out` directory. 

Selective profiling: first start profiling for a large training loop, and once you get an idea about where the bottleneck is, then you can select a few iterations for profiling.

(visualisation-on-a-web-browser)=
### Visualization on a web browser
To view the output data generated from the profiling process, one needs to install TensorBord, which can be done for instance in a virtual environment

- **Step 0** : loading a Python model, creating and activating a virtual environment.
Load a Python module. e.g.: `module` load python/3.9.6-GCCcore-11.2.0`
- `mkdir Myenv`
- `python –m venv Myenv`
- `source Myenv/bin/activate`

- **Step 1**: Installing TensorBoard Plugin via pip wheel packages using the following command (see also [here](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)):
- `python –m pip install torch_tb_profiler`

- **Step 2**: Running Tensorboard using the command:

```console
tensorboard --logdir=./out --bind_all
``` 
will generate a local address having a specific registered or private port, as shown in {ref}`Figure<fig-tensorboard>`. Note that in HPC systems, direct navigation to the generated address is blocked by firewalls. Therefore, connecting to an internal network from outside can be done 
via a mechanism called [local port forwarding](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding). As stated in the [SSH documentation](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding) “Local forwarding is used to forward a port from the client machine to the server machine”.

The syntax for local forwarding, which is configured using the option `–L`, can be written as, e.g.:

```console
ssh -L 6009:local.host:6006 username@server.address.com
```
This syntax enables opening a connection to the jump server `username@server.address.com`, and forwarding any connection from port 6009 on the local machine to port 6006 on the server `username@server.address.com`. 

Lastly, the local address `http://localhost:6009/` can be viewed in a Chrome or Firefox browser.

```{eval-rst}

.. _fig-tensorboard:

.. figure:: pytorch_profiler/Figs/fig0.png
   :width: 600px
   :align: center

   Fig 2: Output of running the tensorboar command `tensorboard --logdir=./out --bind_all`.

```

(performance-metrics)=
### Performance metrics 

In this section, we provide screenshots of different views of performance metrics stemming from PyTorch Profiler. The metrics include:

- GPU usage (cf. {ref}`Figure 3<fig-overview>`)
- GPU Kernel view (cf. {ref}`Figure 4<fig-kernel>`)
- Trace view (cf. {ref}`Figure 5<fig-trace1>` and {ref}`Figure 6<fig-trace2>`)
- Memory view (cf. {ref}`Figure 7<fig-memory>`)
- Module view (cf. {ref}`Figure 8<fig-module>`)


```{eval-rst}

.. _fig-overview:

.. figure:: pytorch_profiler/Figs/fig1.png
   :width: 600px
   :align: center

   Fig 3: Overview of GPU activities.

```

```{eval-rst}
.. _fig-kernel:

.. figure:: pytorch_profiler/Figs/fig2.png
   :width: 600px
   :align: center

   Fig 4: View of GPU Kernels.

```

```{eval-rst}

.. _fig-trace1:

.. figure:: pytorch_profiler/Figs/fig3.png
   :width: 600px
   :align: center

   Fig 5: View of Trace.

```

```{eval-rst}

.. _fig-trace2:

.. figure:: pytorch_profiler/Figs/fig4.png
   :width: 600px
   :align: center

   Fig 6: View of Trace.

```

```{eval-rst}

.. _fig-memory:

.. figure:: pytorch_profiler/Figs/fig5.png
   :width: 600px
   :align: center

   Fig 7: View of Memory usage.

```

```{eval-rst}

.. _fig-module:

.. figure:: pytorch_profiler/Figs/fig6.png
   :width: 600px
   :align: center

   Fig 8: View of Modules.

```


(launching-a-pytorch-based-application)=
## Launching a PyTorch-based application 

For completeness, we provide an example of a job script that incorporates a PyTorch singularity container. The script can be adapted according to the selective computing resources.

```bash
#!/bin/bash -l
#SBATCH --job-name=PyTprofiler
#SBATCH --account=<project_account>
#SBATCH --time=00:10:00     #wall-time 
#SBATCH --partition=accel   #partition 
#SBATCH --nodes=1           #nbr of nodes
#SBATCH --ntasks=1          #nbr of tasks
#SBATCH --ntasks-per-node=1  #nbr of tasks per nodes (nbr of cpu-cores)
#SBATCH --cpus-per-task=1   #nbr of threads
#SBATCH --gpus=1            #total nbr of gpus
#SBATCH --gpus-per-node=1   #nbr of gpus per node
#SBATCH --mem=4G            #main memory
#SBATCH -o PyTprofiler.out

# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

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

(conclusion)=
# Conclusion
In conclusion, we have provided a guide on how to perform code profiling of GPU-accelerated Deep Learning models using the PyTorch Profiler. The particularity of the profiler relies on its simplicity and ease of use without installing additional packages and with a few lines of code to be added. These lines of code constitute the setting of the profiler, which can be customized according to the desired performance metrics. The profiler provides an overview of metrics; this includes a summary of GPU usage and Tensor cores usage (if it is enabled), this is in addition to an advanced analysis-based view of GPU kernel, memory usage in time, trace and modules. These features are key elements for identifying bottlenecks in an application. Identifying these bottlenecks has the benefit of optimizing the application to run efficiently and reliably on HPC systems. 


# Relevant links

[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

[NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

[Local port forwarding](https://www.ssh.com/academy/ssh/tunneling-example#local-forwarding)

[Slides](https://github.com/HichamAgueny/Profiling-GPU-accelerated-DL)

[PyTorch Profiler video](https://www.youtube.com/watch?v=m6ouC0XMYnc&ab_channel=PyTorch)



