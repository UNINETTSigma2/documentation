(byo-cuda-container)=
# BYO CUDA environment with containers

```{note}
This example assumes basic knowledge on Singularity containers,
on the level presented in our container {ref}`intro <running-containers>`.
```

This example demonstrates:

1. how to pull a CUDA container image from the NGC
2. how to build your CUDA application through the container environment
3. how to run your CUDA application through the container environment

In this example we will use the same code as we used in the basic CUDA
{ref}`tutorial <cuda-c>`, but we will try to build the code in a different
CUDA environment than what we currently have available on Saga. In particular,
we will fetch a container image with a _newer_ version of CUDA from the Nvidia GPU
Cloud ([NGC](https://catalog.ngc.nvidia.com/)).

First, let's revisit the source code from the other example:

```{eval-rst} 
.. literalinclude:: cuda/vec_add_cuda.cu
  :language: c
```

```{eval-rst} 
:download:`vec_add_cuda.cu <./cuda/vec_add_cuda.cu>`
```


## Step 1: Pull container image from the NGC

We start by browsing the [NGC](https://catalog.ngc.nvidia.com/containers) for a suitable
container image, which should be from the [CUDA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)
collection. For development work (compilation etc) we should choose a tag with `devel`
in its name, but the CUDA version and operating system can be whatever you like.
We download one such container with the following command (this might take a few minutes):

```console
[me@login.SAGA]$ singularity pull docker://nvcr.io/nvidia/cuda:11.4.0-devel-ubuntu20.04
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
Getting image source signatures
Copying blob 35807b77a593 done  
Copying blob 2f02693dc068 done  
Copying blob 903c09d5b94e done  
Copying blob 205c053b80d7 done  
Copying blob 3da463f4fa89 done  
Copying blob 6ae79230f62a done  
Copying blob 43b3e972ee6d done  
Copying blob 93f128a4f293 done  
Copying blob c8078b8bb166 done  
Copying config c3f63d2c90 done  
Writing manifest to image destination
Storing signatures
2021/11/03 22:40:19  info unpack layer: sha256:35807b77a593c1147d13dc926a91dcc3015616ff7307cc30442c5a8e07546283
2021/11/03 22:40:20  info unpack layer: sha256:2f02693dc0685e3a6de01df36887f5d358f48a48886e688251ac3ef04c410362
2021/11/03 22:40:20  info unpack layer: sha256:903c09d5b94ea239fc1a5a7fd909c3afe62912ce90c86c8924d57a2f71055d34
2021/11/03 22:40:21  info unpack layer: sha256:205c053b80d7b029905054aac222afacbb17ec8266df623e0bcea36ce5d88d37
2021/11/03 22:40:21  info unpack layer: sha256:3da463f4fa89a36aa543d72065de871d22072cd139e6e85b2fb7bd91473a4409
2021/11/03 22:40:21  info unpack layer: sha256:6ae79230f62a71f8abb1c6aaefbaa48e6cf6f38a671ba511bf19678882d747c2
2021/11/03 22:40:43  info unpack layer: sha256:43b3e972ee6de26010ac81c65aa5f37612caa6c1f0c9eb9c114f841f67e154a3
2021/11/03 22:40:43  info unpack layer: sha256:93f128a4f293c7f83d182fda8740bb51ea5c1c7508c97f6b563e08c12c3fca07
2021/11/03 22:41:12  info unpack layer: sha256:c8078b8bb1668a188a782de56e7d1e6faff012f45a932f1c43b145c2b61ea0d3
INFO:    Creating SIF file...
```

We should now have the following file in the current directory

```console
[me@login.SAGA]$ ls -lh
-rwxrwxr-x 1 me me_g 2.8G Nov  3 12:41 cuda_11.4.0-devel-ubuntu20.04.sif
```

which is the Singularity image file. Notice the size (2.8G) of this image, and keep in
mind your `$HOME` disk quota of 20GiB.

Once we have the container image we can verify that we have the `nvcc` CUDA compiler
available inside it. First we check on the host (no modules load at this point):

```console
[me@login.SAGA]$ nvcc --version
-bash: nvcc: command not found
```

but if we run the same command _through_ the container we should find a compiler:

```console
[me@login.SAGA]$ singularity exec cuda_11.4.0-devel-ubuntu20.04.sif nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Aug_15_21:14:11_PDT_2021
Cuda compilation tools, release 11.4, V11.4.120
Build cuda_11.4.r11.4/compiler.30300941_0
```

Notice the CUDA version, which is more recent than any currently supported module
on Saga (which is CUDA/11.1.1 at the time of writing).

```{warning}
Container images are typically a few GiB in size, so you might want to keep your
containers in a project storage area to avoid filling up your limited `$HOME` disk quota.
Also beware that pulled images are cached, by default under `$HOME/.singularity/cache`.
This means that if you pull the same image twice, it will be immediately available from
the cache without downloading/building, but it also means that it will consume disk space.
To avoid this you can either add `--disable-cache` to the `pull` command, change the cache
directory with the `SINGULARITY_CACHEDIR` environment variable, or clean up the cache
regularly with `singularity cache clean`.
```

## Step 2: Compile the code through the container environment

Now that we found a `nvcc` compiler inside the container, we can try to compile our code
_through_ the container environment. In order to make this work we need to add one option
to the compilation command that we used {ref}`before <cuda-c>`. In the "native" build on
Saga we did not have to specify the `gpu-architecture` that we compile for, as it was able
to pick that up automatically. In the container environment, however, the CUDA compiler
has not been set up with this information so we have to provide it as a compiler option,
which makes the full compilation string:

```
nvcc --gpu-architecture=sm_60 vec_add_cuda.cu -o vec_add_cuda
```

and we just have to pass it to the `singularity exec` command:

```console
[me@login.SAGA]$ singularity exec --bind $PWD cuda_11.4.0-devel-ubuntu20.04.sif nvcc --gpu-architecture=sm_60 vec_add_cuda.cu -o vec_add_cuda
```

Notice also the `--bind $PWD` option, which makes sure that the files in the current directory
is visible to the container. The compilation should hopefully finish without any errors/warnings,
and there should now be a `vec_add_cuda` executable file in your current directory (also
accessible by the host, even if it was created from within the container).

This executable will however not run successfully on the login node, since there are no
GPUs available here. We thus have to request GPU resources through Slurm.

```{note}
The origin of the `sm_60` architecture flag is a bit hard to explain, and you really need
to dig deep into the hardware specs to figure out the correct option here.
For our machines at NRIS we have:
- Saga: Nvidia Pascal (P100) - `sm_60`
- NIRD: Nvidia Volta (V100) - `sm_70`
- Betzy: Nvidia Ampere (A100) - `sm_80`
```

## Step 3: Run the code through the container environment

We will test the code in an interactive session, so we ask for a single GPU:

```console
[me@login.SAGA]$ salloc --nodes=1 --gpus=1 --time=0:10:00 --mem=1G --partition=accel --account=<your-account>
salloc: Pending job allocation 4320527
salloc: job 4320527 queued and waiting for resources
salloc: job 4320527 has been allocated resources
salloc: Granted job allocation 4320527
salloc: Waiting for resource configuration
salloc: Nodes c7-8 are ready for job
```

Now, the CUDA environment inside the container is probably backward compatible with the
CUDA version on the cluster, so our newly created executable will _probably_ run smoothly
even if we don't run it through the container, but for the sake of consistency we will
launch the program with `singularity exec`

```console
[me@c7-8]$ singularity exec --bind $PWD cuda_11.4.0-devel-ubuntu20.04.sif ./vec_add_cuda
ENTER MAIN
Segmentation fault
```

The reason this failed is that Singularity has not been made aware of the GPU resources
on the host (kind of like the situation we had with `salloc --partition=accel` _without_
the `--gpus=1` option). The magic keyword to make this work in Singularity is `--nv`
(for Nvidia, for AMD GPUs the keyword is `--rocm`):

```console
[me@c7-8]$ singularity exec --nv --bind $PWD cuda_11.4.0-devel-ubuntu20.04.sif ./vec_add_cuda
ENTER MAIN
c[0]  : 1.000000
c[1]  : 1.000000
c[42] : 1.000000
EXIT SUCCESS
```

We have now successfully compiled and run a CUDA program _through_ a container environment.
