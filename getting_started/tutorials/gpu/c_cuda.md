# Using CUDA in C

This example demonstrates:

1. how to compile a simple CUDA program
2. how to request GPU resources and run the program
3. how to verify that the code actually executed on the GPU device

In this example we will use [CUDA](https://en.wikipedia.org/wiki/CUDA) to
facilitate offloading of a simple vector addition to be performed by a GPU,
and we will try to verify that the code is _actually_ executed on the device.
We will compile and run the following code on Saga:

```{eval-rst} 
.. literalinclude:: vec_add_cuda.cu
  :language: c
```

```{eval-rst} 
:download:`vec_add_cuda.cu <./vec_add_cuda.cu>`
```

```{note}
The purpose of this example is _not_ to understand the details in the code snippet
above, but rather to have a working code example that we can compile, run and verify
on a GPU.
```

## Step 1: Compiling the code

In order to compile this code we need a CUDA-aware compiler, and on Saga we get this
by loading a `CUDA` module (choosing here the most recent version at the time of writing):

```console
[me@login.SAGA]$ module load CUDA/11.1.1-GCC-10.2.0
```

After the module is loaded you should have the `nvcc` CUDA compiler available:

```console
[me@login.SAGA]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

We can now compile the code with the following command (never mind optimization flags
etc, they are not important at this point):

```console
[me@login.SAGA]$ nvcc vec_add_cuda.cu
```

This command should ideally finish is silence. If we try to run the resulting executable `a.out`:

```console
[me@login.SAGA]$ ./a.out
ENTER MAIN
Segmentation fault (core dumped)
```

It will fail because we are still running on the `login` node, and there are no GPU
hardware and drivers available here. The next step is thus to request GPU resources
for running our program.

```{note}
In order to run a CUDA program you must have CUDA hardware drivers installed on the
machine. On Saga, these are _only_ available on the GPU compute nodes, _not_ on the
login nodes. However, the drivers are not necessary for the compilation step (only the
CUDA library, which comes with `module load CUDA/...`), so this can be done on the login node.
```

## Step 2: Running the code

## Step 3: Verifying the run

To run this we will first have to create a Slurm script in which we will request
resources. A good place to start is with a basic job
script (see {ref}`job-scripts`).
Use the following to create `submit_gpu.sh` (remember to substitute your project
number under `--account`):

```{eval-rst} 
.. literalinclude:: submit_cpu.sh
  :language: bash
```
```{eval-rst} 
:download:`submit_gpu.sh <./submit_cpu.sh>`
```

If we just run the above Slurm script with `sbatch submit_gpu.sh` the output
(found in the same directory as you executed the `sbatch` command with a name
like `slurm-<job-id>.out`) will contain several errors as `Tensorflow` attempts
to communicate with the GPU, however, the program will still run and give the
following successful output:

```bash
Num GPUs Available:  0                   
tf.Tensor(                               
[[22. 28.]                               
 [49. 64.]], shape=(2, 2), dtype=float32)
```

So the above, eventually, ran fine, but did not report any GPUs. The reason for
this is of course that we never asked for any GPUs in the first place. To remedy
this we will change the Slurm script to include the `--partition=accel` and
`--gpus=1`, as follows:

```{eval-rst} 
.. literalinclude:: submit_gpu.sh
  :language: bash
  :emphasize-lines: 7,8
```
```{eval-rst} 
:download:`submit_gpu.sh <./submit_gpu.sh>`
```

We should now see the following output:

```bash
Num GPUs Available:  1                    
tf.Tensor(                                
[[22. 28.]                                
 [49. 64.]], shape=(2, 2), dtype=float32) 
```

However, with complicated libraries such as `Tensorflow` we are still not
guaranteed that the above actually ran on the GPU. There is some output to
verify this, but we will check this manually as that can be applied more
generally.


## Monitoring the GPUs

To do this monitoring we will start `nvidia-smi` before our job and let it run
while we use the GPU. We will change the `submit_gpu.sh` Slurm script above to
`submit_monitor.sh`, shown below:

```{eval-rst} 
.. literalinclude:: submit_monitor.sh
  :language: bash
  :emphasize-lines: 19-21,25
```
```{eval-rst} 
:download:`submit_monitor.sh <./submit_monitor.sh>`
```

```{note}
The query used to monitor the GPU can be further extended by adding additional
parameters to the `--query-gpu` flag. Check available options
[here](http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf).
```

Run this script with `sbatch submit_monitor.sh` to test if the output
`gpu_util-<job id>.csv` actually contains some data. We can then use this data
to ensure that we are actually using the GPU as intended. Pay specific attention
to `utilization.gpu` which shows the percentage of how much processing the GPU
is doing. It is not expected that this will always be `100%` as we will need to
transfer data, but the average should be quite high.

