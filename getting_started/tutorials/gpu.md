# Introduction to using GPU compute
A GPU, or **G**raphics **P**rocessing **U**nit, is a computational unit, which
as the name suggest, is optimized to work on graphics tasks. Nearly every
computer device that one interacts with contains a GPU of some sort, responsible
for transforming the information we want to display into actual colors on our
screens.

One question that might immediately present itself is, **if GPUs are optimized
for graphics - why are they interesting in the context of computational resources?** The
answer to that is of course complicated, but the short explanation is that quite
many computational tasks have a lot in common with graphical
computations. The reason for this is that GPUs are optimized for working with
pixels on the screen, and a lot of them. Since all of these operations are
almost identical mainly working on floating point values, they can be run in
parallel on dedicated hardware (i.e. the GPU) that is tailored and optimized for this particular task. This already sounds quite a bit
like working with a discrete grid in e.g. atmospheric simulation, which points
to the reason why GPUs can be interesting in a scientific context.

Since GPUs are optimized for working on grids of data and how to transform this
data, they are quite well suited for matrix calculations. For some indication of
this we can compare the theoretical performance of one GPU with one CPU
.

| | AMD Epyc 7742 (Betzy) | Nvidia P100 (Saga) | Nvidia A100 |
|-|-----------------------|--------------------|-------------|
| Half Precision | N/A | 18.7 TFLOPS | 78 TFLOPS |
| Single Precision | 1,3 TFLOPS | 9,3 TFLOPS | 19.5 TFLOPS |
| Double Precision | N/A | 4.7 TFLOPS | 9.7 TFLOPS |

Based on this it is no wonder why tensor libraries such as
[`TensorFlow`](https://www.tensorflow.org/) and [`PyTorch`](https://pytorch.org/)
[report **speedup**](https://blog.tensorflow.org/2018/04/speed-up-tensorflow-inference-on-gpus-tensorRT.html)
on accelerators between **`23x` and `190x`** compared to using only a CPU.

## Getting started
Of the resources provided by us, only
[Saga](https://documentation.sigma2.no/jobs/job_types/saga_job_types.html#job-type-saga-accel)
currently has GPUs available. To access these one has to select the correct
partition as well as request one or more GPUs to utilize.

To select the correct partition use the `--partition=accel` flag with either
[`srun`](https://documentation.sigma2.no/jobs/interactive_jobs.html) or in your
[Slurm script](https://documentation.sigma2.no/jobs/job_scripts.html). This flag
will ensure that your job is only run on machines in the `accel` partition which
have attached GPUs. However, to be able to actually interact with one or more
GPUs we will have to also add `--gres=gpu:N` which tells Slurm that we would
also like to use `N` GPUs (`N` can be a number between 1 and 4 on Saga).

### Step by step
To get started we first have to [`SSH` into
Saga](https://documentation.sigma2.no/getting_started/create_ssh_keys.html).

```bash
$ ssh <username>@saga.sigma2.no
```

#### Interactive testing
All projects should have access to GPU resources, and to that end we will start
by simply testing that we can get access to a single GPU. To do this we will run
an interactive job, on the `accel` partition and asking for a single GPU.

```bash
$ srun --ntasks=1 --mem-per-cpu=1G --time=00:02:00 --partition=accel --gres=gpu:1 --qos=devel --account=<your project number> --pty bash -i
$ nvidia-smi
```

The two commands above should result in something like:

```bash
Tue Mar 23 14:29:33 2021                                                       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.32.00    Driver Version: 455.32.00    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:14:00.0 Off |                    0 |
| N/A   33C    P0    30W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

```{note}
In the above Slurm specification we combined `--qos=devel` with GPUs and
interactive operations so that we can experiment with commands interactively.
This can be a good way to perform short tests to ensure that libraries correctly
pick up GPUs when developing your experiments.
```

#### Slurm script testing
The next thing that we will try to do is to utilize the
`TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4` library to execute a very simple
computation on the GPU. We could do the following interactively in Python, but
to introduce both interactive `srun` and Slurm scripts we will now transition to
a Slurm script (which can also make it a bit easier since we don't have to sit
and wait for the interactive session to start).

We will use the following simple calculation in Python and `Tensorflow` to test
the GPUs of Saga (save the following in `gpu_into.py`).

```python
#!/usr/bin/env python 3

import tensorflow as tf

# Test if there are any GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Have Tensorflow output where computations are run
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

# Print result
print(c)
```

To run this we will first have to create a Slurm script in which we will request
resources. A good place to start is with [a basic job
script](https://documentation.sigma2.no/jobs/job_scripts.html#job-script-basics).
Use the following to create `submit_gpu.sh` (remember to substitute your project
number under `--account`):

```bash
#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=512M

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
module list

python gpu_intro.py
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
`--gres=gpu:1`, as follows:

```{eval-rst} 
.. literalinclude:: gpu/submit_gpu.sh
  :language: bash
  :emphasize-lines: 6,7
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

#### Monitoring the GPUs
To do this monitoring we will start `nvidia-smi` before our job and let it run
while we use the GPU. We will change the `submit_gpu.sh` Slurm script above to
`submit_monitor.sh`, shown below:

```{eval-rst} 
.. literalinclude:: gpu/submit_monitor.sh
  :language: bash
  :emphasize-lines: 18-20,24
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
transfer data back and forth, but the average should be quite high.

## Next steps
Transitioning your application to GPU can be a daunting challenge. We have
documented [a few ways to get
started](https://documentation.sigma2.no/code_development/guides.html), but if
you are unsure please don't hesitate to contact us at
[support@metacenter.no](mailto:support@metacenter.no).

We also have a few tutorials on specific libraries:
- [Tensorflow](https://documentation.sigma2.no/jobs/guides/tensorflow_gpu.html)
- [OpenACC](https://documentation.sigma2.no/code_development/guides/openacc.html)
- Coming soon
  - OpenMP for GPU
