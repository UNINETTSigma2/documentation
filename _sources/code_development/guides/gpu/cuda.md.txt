(cuda-c)=
# Using CUDA in C

This example demonstrates:

1. how to compile a simple CUDA program
2. how to request GPU resources and run the program
3. how to monitor the GPU utilization

In this example we will use [CUDA](https://en.wikipedia.org/wiki/CUDA) to
facilitate offloading of a simple vector addition to be performed by a GPU,
and we will try to verify that the code is _actually_ executed on the device.
We will compile and run the following code on Saga:

```{eval-rst} 
.. literalinclude:: cuda/vec_add_cuda.cu
  :language: c
```

```{eval-rst} 
:download:`vec_add_cuda.cu <./cuda/vec_add_cuda.cu>`
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
[me@login.SAGA]$ nvcc vec_add_cuda.cu -o vec_add_cuda
```

This command should hopefully finish without any error/warning. We can try to run the
resulting executable (which we called `vec_add_cuda`):

```console
[me@login.SAGA]$ ./vec_add_cuda
ENTER MAIN
Segmentation fault (core dumped)
```

But it will fail because we are here still running on the `login` node, and there are
no GPU hardware and drivers available here. The next step is thus to request GPU resources
for running our program.

```{note}
In order to run a CUDA program you must have CUDA hardware drivers installed on the
machine. On Saga, these are _only_ available on the GPU compute nodes, _not_ on the
login nodes. However, the drivers are not necessary for the compilation step (only the
CUDA library, which comes with `module load CUDA/...`), so this can be done on the login node.
```

## Step 2: Running the code

We will first test the code in an interactive session, so we ask for a single GPU:

```console
[me@login.SAGA]$ salloc --nodes=1 --gpus=1 --time=0:10:00 --mem=1G --partition=accel --account=<your-account>
salloc: Pending job allocation 4320527
salloc: job 4320527 queued and waiting for resources
salloc: job 4320527 has been allocated resources
salloc: Granted job allocation 4320527
salloc: Waiting for resource configuration
salloc: Nodes c7-8 are ready for job
```

Remember to load the `CUDA` module if not already loaded from Step 1. You can also verify
that you actually have access to a GPU using the `nvidia-smi` command. If all goes well,
your program should now  run and exit successfully:

```console
[me@c7-8]$ ./vec_add_cuda
ENTER MAIN
c[0]  : 1.000000
c[1]  : 1.000000
c[42] : 1.000000
EXIT SUCCESS
```

We here see the expected output of $c[i] = sin^2(i) + cos^2(i) = 1$ for any $i$, which
means that the code runs correctly.

```{note}
For this particular example we have actually now already verified that the code was executed
**on the GPU**. As the code is written, there is no "fallback" implementation that runs
on the CPU in case no GPU is found, which means that `EXIT SUCCESS` == "the code executed
on the GPU".
```

## Step 3: Monitor the GPU utilization

We will now try to capture some stats from the execution using the `nvidia-smi` tool
to verify that we were able to utilize a few percent of the GPUs capacity. To get a
reasonable reading from this tool we need an application that runs for at least a few
seconds, so we will first make the following change to our source code:

```{eval-rst} 
.. literalinclude:: cuda/loop_add_cuda.cu
  :language: c
  :lines: 41-46
  :emphasize-lines: 1,6
```
```{eval-rst} 
:download:`loop_add_cuda.cu <./cuda/loop_add_cuda.cu>`
```

i.e. we loop over the vector addition 100 000 times. This should hopefully give sufficient
run time to be picked up by our tool. We then compile and run our new code with the following
job script:

```{eval-rst} 
.. literalinclude:: cuda/run.sh
  :language: bash
```
```{eval-rst} 
:download:`run.sh <./cuda/run.sh>`
```

Submit the job using `sbatch` (remember to set the `--account` option, and note that we are
back on the `login` node):

```console
[me@login.SAGA]$ sbatch run.sh
Submitted batch job 4320512
```

Wait for the job to finish and verify from the `slurm-xxxxx.out` file that the
calculation still finished successfully, and that it ran for at least a few seconds.

We can then add the following lines to the script in order to monitor the GPU
utilization using `nvidia-smi`:

```{eval-rst} 
.. literalinclude:: cuda/monitor.sh
  :language: bash
  :lines: 17-32
  :emphasize-lines: 4-7,12-13
```
```{eval-rst} 
:download:`monitor.sh <./cuda/monitor.sh>`
```

Submit the job:

```console
[me@login.SAGA]$ sbatch monitor.sh
Submitted batch job 4320513
```

Wait for the job to complete and inspect the `monitor-xxxx.csv` file we just created:

```console
[me@login.SAGA]$ cat monitor-4320513.csv
timestamp, utilization.gpu [%], utilization.memory [%]
2021/11/03 21:42:44.210, 0 %, 0 %
2021/11/03 21:42:45.211, 82 %, 76 %
2021/11/03 21:42:46.211, 82 %, 69 %
2021/11/03 21:42:47.211, 82 %, 69 %
```

We see here that the GPU utilization reached 82% of the GPUs capacity.
