# Using TensorFlow in Python

In this example we will try to utilize the
`TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0` library to execute a very simple
computation on the GPU. We could do the following interactively in Python, but
we will instead use a Slurm script, which will make it a bit more reproducible
and in some sense a bit easier, since we don't have to sit and wait for the
interactive session to start.

We will use the following simple calculation in Python and `TensorFlow` to test
the GPUs:

```{eval-rst} 
.. literalinclude:: tensorflow/gpu_intro.py
  :language: python
```

```{eval-rst} 
:download:`gpu_intro.py <./tensorflow/gpu_intro.py>`
```

To run this we will first have to create a Slurm script in which we will request
resources. A good place to start is with a basic job
script (see {ref}`job-scripts`).
Use the following to create `submit_cpu.sh` (remember to substitute your project
number under `--account`):

`````{tabs}
````{group-tab} Saga

```{eval-rst} 
.. literalinclude:: tensorflow/submit_cpu.sh
  :language: bash
```
```{eval-rst} 
:download:`submit_cpu.sh <./tensorflow/submit_cpu.sh>`
```
````
`````

If we just run the above Slurm script with `sbatch submit_cpu.sh` the output
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
this is of course that we never asked for any GPUs in the first place. To remedy this we will include the `--gpus=1` in the Slurm script. We must also specify which partition we want to use (`--partition=accel` for P100 and `--partition=a100` for A100):

`````{tabs}
````{group-tab} Saga (P100)

```{eval-rst} 
.. literalinclude:: tensorflow/submit_gpu.sh
  :language: bash
  :emphasize-lines: 7,8
```
```{eval-rst} 
:download:`submit_gpu.sh <./tensorflow/submit_gpu.sh>`
```
````
````{group-tab} Saga (A100)

```{eval-rst}
.. literalinclude:: tensorflow/submit_gpu_a100.sh
  :language: bash
  :emphasize-lines: 7,8,14
```
 ```{eval-rst}
:download:`submit_gpu_a100.sh <./tensorflow/submit_gpu_a100.sh>`
```
````

`````

```{note}
To use the A100 GPUs (second tab above) we must include `module --force swap StdEnv Zen2Env`
 (see {ref}`this page <building-gpu>` for an explanation).
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

`````{tabs}
````{group-tab} Saga

```{eval-rst} 
.. literalinclude:: tensorflow/submit_monitor.sh
  :language: bash
  :emphasize-lines: 19-21,25
```
```{eval-rst} 
:download:`submit_monitor.sh <./tensorflow/submit_monitor.sh>`
```
````
`````

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

