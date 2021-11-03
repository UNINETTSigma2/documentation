# Introduction to using GPU compute

A GPU, or **G**raphics **P**rocessing **U**nit, is a computational unit, which
as the name suggest, is optimized to work on graphics tasks. Nearly every
computer device that one interacts with contains a GPU of some sort, responsible
for transforming the information we want to display into actual pixels on our
screens.

One question that might immediately present itself is, **if GPUs are optimized
for graphics - why are they interesting in the context of computational
resources?** The answer to that is of course complicated, but the short
explanation is that many computational tasks have a lot in common with
graphical computations. The reason for this is that GPUs are optimized for
working with pixels on the screen, and a lot of them. Since all of these
operations are almost identical, mainly working on floating point values, they
can be run in parallel on dedicated hardware (i.e. the GPU) that is tailored and
optimized for this particular task. This already sounds quite a bit like working
with a discrete grid in e.g. atmospheric simulation, which points to the reason
why GPUs can be interesting in a computational context.

Since GPUs are optimized for working on grids of data and how to transform this
data, they are quite well suited for matrix calculations. For some indication of
this we can compare the theoretical performance of one GPU with one CPU
.

| | AMD Epyc 7742 (Betzy) | Nvidia P100 (Saga) | Nvidia A100 (Betzy)|
|-|-----------------------|--------------------|-------------|
| Half Precision | N/A | 18.7 TFLOPS | 78 TFLOPS |
| Single Precision | 1,3 TFLOPS | 9,3 TFLOPS | 19.5 TFLOPS |
| Double Precision | N/A | 4.7 TFLOPS | 9.7 TFLOPS |

Based on this it is no wonder why tensor libraries such as
[`TensorFlow`](https://www.tensorflow.org/) and [`PyTorch`](https://pytorch.org/)
[report **speedup**](https://blog.tensorflow.org/2018/04/speed-up-tensorflow-inference-on-gpus-tensorRT.html)
on accelerators between **`23x` and `190x`** compared to using only a CPU.


## Getting started

To get started we first have to {ref}`ssh` into Saga:
```console
[me@mylaptop]$ ssh <username>@saga.sigma2.no
```

From the {ref}`hardware specification <saga>` we see that there should be 8 GPU
nodes available on Saga, and from the available {ref}`job types <job_type_saga_accel>`
we identify `--partition=accel` as the relevant hardware partition for GPU jobs.
You can run the `sinfo` command to check the available partitions on Saga:

```console
[me@login.SAGA]$ sinfo
```
```{eval-rst}
.. literalinclude:: gpu/sinfo.out
    :emphasize-lines: 9-11
```

Here we see that the `accel` partition contains 8 nodes in total, 2 of which are
unused at the moment (`idle`), 4 are fully occupied (`alloc`) and 2 are partially
occupied (`mix`). We can also read from this that the maximum time limit for a GPU
job is 14 days, which might be relevant for your production calculations.

To select the correct partition use the `--partition=accel` flag with either
`salloc` ({ref}`interactive <interactive-jobs>`)
or
`sbatch` ({ref}`job script <job-scripts>`).
This flag will ensure that your job is only run on machines in the `accel` partition
which have attached GPUs. However, to be able to actually interact with one or more
GPUs we will have to also add `--gpus=N` which tells Slurm that we would also like
to use `N` GPUs (`N` can be a number between 1 and 4 on Saga since each node has 4
GPUs).

```{tip}
There are multiple ways of requesting GPUs a part from `--gpus=N`, such as
`--gpus-per-task` to specify the number of GPUs that each task should get
access to. Checkout the official [Slurm
documentation](https://slurm.schedmd.com/srun.html) for more on how to specify
the number of GPUs.
```


## Interactive testing

All projects should have access to GPU resources, and to that end we will start
by simply testing that we can get access to a single GPU. To do this we will run
an interactive job using the `salloc` command, on the `accel` partition and asking
for a single GPU:

```console
[me@login.SAGA]$ salloc --ntasks=1 --mem-per-cpu=1G --time=00:02:00 --partition=accel --gpus=1 --qos=devel --account=<your project number>
salloc: Pending job allocation 4318997
salloc: job 4318997 queued and waiting for resources
salloc: job 4318997 has been allocated resources
salloc: Granted job allocation 4318997
salloc: Waiting for resource configuration
salloc: Nodes c7-7 are ready for job
```

Once we land on the compute node we can inspect the GPU hardware with
the `nvidia-smi` command (this is kind of the `top` equivalent for Nvidia GPUs):

```console
[me@c7-8.SAGA]$ nvidia-smi
```
```{eval-rst}
.. literalinclude:: gpu/nvidia-smi.out
    :emphasize-lines: 3,9,19
```

Here we can find useful things like CUDA library/driver version and the name of the
graphics card (`Tesla P100-PCIE...`), but also information about currently
running processes that are "GPU aware" (none at the moment). If you don't get any
useful information out of the `nvidia-smi` command (e.g. `command not found` or
`No devices were found`) you likely missed the `--partition=accel` and/or `--gpus=N`
options in your Slurm command, which means that you won't actually have access to any
GPU (even if there might be one physically on the machine).

```{tip}
In the above Slurm specification we combined `--qos=devel` with GPUs and
interactive operations so that we can experiment with commands interactively.
This can be a good way to perform short tests to ensure that libraries correctly
pick up GPUs when developing your experiments. Read more about `--qos=devel`
in our guide on {ref}`interactive jobs <interactive-jobs>`.
```

## Simple GPU test runs

```{eval-rst}
.. toctree::
    :maxdepth: 1

    gpu/python_tensorflow.md
    gpu/c_cuda.md
```


## Next steps

Transitioning your application to GPU can be a daunting challenge. We have
documented a few ways to get started in our development {ref}`guides <dev-guides>`,
but if you are unsure please don't hesitate to contact us at
[support@nris.no](mailto:support@nris.no).

We also have a few tutorials on specific GPU related topics:
- {ref}`openacc`
- {ref}`hipsycl-start`
- {ref}`Running TensorFlow on GPUs <tensorflow>`
- {ref}`Running containers w/CUDA support: BigDFT example <bigdft-cuda-example>`
