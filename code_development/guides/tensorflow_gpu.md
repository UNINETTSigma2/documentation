---
orphan: true
---

```{index} GPU; TensorFlow on GPU, TensorFlow; TensorFlow on GPU
```

(tensorflow)=

# TensorFlow on GPU
This guide provides an introduction to use machine learning libraries on GPU.
The examples ask for Sigma2 recources with GPU. This is achieved with the `--partition=accel --gpus=1`({ref}`job_scripts_saga_accel`).



```{note}
When loading modules pressing `<tab>` gives you
autocomplete options.
```

```{note}
It can be useful to do an initial `module reset` to
ensure nothing from previous experiments is loaded before
loading modules for the first time.
```

```{note}
Modules are regularly updated so if you would like a newer
version, than what is listed above, use `module avail |
less` to browse all available packages.
```

Below we will go through the main approaches for using TensorFlow:

- Using the preinstalled TensorFlow module
- Loading TensorFlow from EESSI
- Using a container

## Using the preinstalled TensorFlow module

This is the easiest and recommended option for most users. The preinstalled modules are optimized for the cluster hardware, pre-configured with correct dependencies (cuDNN, NCCL, etc.), and tested to work with the available GPU drivers.

### Finding available versions

To discover which TensorFlow versions are installed:

```bash
# Show all available versions with descriptions
$ module spider TensorFlow

# Show versions you can load directly
$ module avail TensorFlow

# Show dependencies loaded by a specific module
$ module show TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Load an available TensorFlow version
$ module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
```

```{note}
Modules with `CUDA` in the name are GPU-enabled. Modules without `CUDA` run on CPU only.
```

## EESSI

[EESSI (European Environment for Scientific Software Installations)](https://eessi.io/docs/) provides a curated collection of scientific software that is built once and distributed globally via CernVM-FS. Software is optimized for different CPU architectures and works seamlessly across systems.


## Loading TensorFlow from EESSI

First, load the EESSI environment:

```bash
$ module load EESSI/2023.06
```

Then load TensorFlow:

```bash
$ module load TensorFlow/2.13.0-foss-2023a
```

To see all available TensorFlow versions in EESSI:

```bash
$ module spider TensorFlow
```

For more information, see {ref}`EESSI documentation <eessi>`.

## Containers

If you need a specific TensorFlow version not available as a module or via EESSI, you can use a container. This gives you complete control over the software environment.

### Using Apptainer/Singularity

The clusters support **Apptainer/Singularity**, which can run Docker images in HPC environments. For GPU workloads, NVIDIA provides optimized TensorFlow containers on the [NGC catalog][tensorflow_containers].

```{tip}
When choosing a container, ensure it supports:
- GPU (look for tags with `gpu` or check the description)
- The `amd64` architecture (x86_64)
```

### Pulling a container

```bash
# Pull a TensorFlow container from NVIDIA NGC
$ apptainer pull docker://nvcr.io/nvidia/tensorflow:23.09-tf2-py3
```

This creates a `.sif` file (Singularity Image Format) in your current directory.

```{warning}
Container images can be large (several GB). Store them in your project area rather than your home directory to avoid quota issues.
```

### Verifying the container

```bash
# Check TensorFlow version
$ apptainer run tensorflow_23.09-tf2-py3.sif python -c "import tensorflow as tf; print(tf.__version__)"
```

### Running jobs with containers

In your Slurm job script, use `apptainer exec` with the `--nv` flag to enable GPU support:

```bash
apptainer exec --nv \
    --bind /cluster/projects/nnXXXXk:/cluster/projects/nnXXXXk \
    tensorflow_23.09-tf2-py3.sif python mnist.py
```

Key options:
- `--nv`: Enables NVIDIA GPU support inside the container
- `--bind`: Mounts directories from the host system into the container

For more details on containers, see {ref}`Running containers <running-containers>`.


## Loading data

There are two ways to load data:

- Uploading data from your local machine using `rsync`
- Loading built-in datasets

### Loading data using `rsync`

Use `rsync` to upload data from your computer:

```bash
rsync -zh --info=progress2 -r /path/to/dataset/folder <username>@saga.sigma2.no:~/.
```

For large amounts of data it is recommended to load into your {ref}`project-area`
to avoid filling your home area.

To retrieving the path to the dataset, we can utilize python's [`os`][python_os]
module to access the variable, like so:

```python
# ...Somewhere in your experiment python file...
# Load 'os' module to get access to environment and path functions
import os

# Path to dataset
dataset_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], 'dataset')
```

If you use a container, you must define your dataset path like this:

```python
dataset_path = os.path.join(os.environ['PWD'], 'dataset')
```


### Loading built-in datasets
First we will need to download the dataset on the login node. Ensure that the
correct modules are loaded. Next open up an interactive python session with
`python`, then:

```python
tf.keras.datasets.mnist.load_data()
```

This will download and cache the MNIST dataset which we can use for training
models. Load the data in your training file like so:

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
```
 

## Saving model data
For saving model data and weights we suggest the `TensorFlow` [built-in
checkpointing and save functions][tensorflow_ckpt].

Below follows an example of how to load built-in data and save weights.

```{eval-rst}
.. literalinclude:: files/mnist_save_model_data.py
  :language: python
  
```

The Python script above can be run with the following job script:

```{eval-rst}
.. literalinclude:: files/mnist_test.sh
  :language: bash
  
```


Once these two files are located on a Sigma2 resource we can run it with:

```bash
$ sbatch mnist_test.sh
```

In your code it is important to load the latest checkpoint if
available, which can be retrieved with:

```python
# Load weights from a previous run
ckpt_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
			"<job_id to load checkpoints from>",
			"checkpoints")
latest = tf.train.latest_checkpoint(ckpt_dir)

# Create a new model instance
model = create_model()

# Load the previously saved weights if they exist
if latest:
	model.load_weights(latest)
```


## Using `TensorBoard`

[`TensorBoard`][tensorboard] is a nice utility for comparing different runs and viewing progress during optimization. To enable this on Sigma2 resources we will need to write data into our home area and some steps are necessary for connecting and viewing the board.

We will continue to use the MNIST example from above. The following changes are needed to enable `TensorBoard`.

```python
# In the 'mnist.py' script
import datetime

# We will store the 'TensorBoard' logs in the folder where the 'mnist_test.sh'
# file was launched and create a folder like 'logs/fit'. In your own code we
# recommended that you give these folders names that you will recognize,
# the last folder uses the time when the program was started to separate related
# runs
log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
		       "logs",
		       "fit",
		       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Change the last line, where we fit our data in the example above, to also
# include the TensorBoard callback
model.fit(x_train[:1000], y_train[:1000],
          epochs=50,
	  # Change here:
          callbacks=[ckpt_callback, tensorboard_callback],
          validation_data=(x_test[:1000], y_test[:1000]),
          verbose=0)
```

Once you have started a job with the above code embedded, or have a previous run
which created a `TensorBoard` log, it can be viewed as follows.

1. Copy the log files into your local machine and ensure `TensorBoard` is installed.
2. Run `tensorboard --logdir=/path/to/logs/fit --port=0`.
3. In the output from the above command note which port `TensorBoard` has
   started on, the last line should look something like: `TensorBoard 2.1.0 at
   http://localhost:44124/ (Press CTRL+C to quit)`.
4. Click the link provided in the output to view the results.



## Advanced topics

### Using multiple GPUs

Since all of the GPU machines on Saga have four GPUs it can be beneficial for some workloads to distribute the work over more than one device at a time. This can be accomplished with the `tf.distribute.MirroredStrategy`.

```{eval-rst}
.. literalinclude:: files/mnist_multiple_gpus.py
  :language: python
  :emphasize-lines: 11-13, 25-29, 44-46
  
```


To ask for two GPUs, modify the slurm script:
```{eval-rst}
.. literalinclude:: files/mnist_test_two_gpus.sh
  :language: bash
  :emphasize-lines: 8
  
```

### Distributed training on multiple nodes

To utilize more than four GPUs we will turn to the [`Horovod`][hvd] project
which supports several different machine learning libraries and is capable of
utilizing `MPI`. `Horovod` is responsible for communicating between different
nodes and perform gradient computation, averaged over the different nodes.

The following example demonstrates how to adapt the MNIST training script for
multi-node training with `Horovod`. Key modifications include initializing
`Horovod`, pinning each MPI rank to a specific GPU, scaling the learning rate
with the number of workers, and using `Horovod`'s distributed optimizer and
callbacks for gradient averaging and synchronized checkpointing:

```{eval-rst}
.. literalinclude:: files/mnist_hvd.py
  :language: python
  
```

The following SLURM script is designed to run a distributed training job using 
`Horovod` and TensorFlow on Saga. It requests 2 nodes, each with 4 GPUs, and distributes the
 workload across 8 tasks (1 task per GPU). The script also ensures that the necessary modules and environment variables are properly configured.

```{eval-rst}
.. literalinclude:: files/mnist_hvd.sh
  :language: bash
  
```

```{note}
To use a100 GPUs on Saga, replace `--partition=accel` with `--partition=a100` and add  `module --force swap StdEnv Zen2Env` to your jobscript
 (more information {ref}`here <building-gpu>`).
```


[sigma2]: https://www.sigma2.no/
[tensorflow]: https://www.tensorflow.org/
[virtualenv]: https://virtualenv.pypa.io/en/stable/
[tf_install]: https://www.tensorflow.org/install/pip
[rsync]: https://en.wikipedia.org/wiki/Rsync
[python_os]: https://docs.python.org/3/library/os.html
[tensorflow_ckpt]: https://www.tensorflow.org/tutorials/keras/save_and_load
[tensorboard]: https://www.tensorflow.org/tensorboard
[mirrored]: https://www.tensorflow.org/guide/distributed_training#mirroredstrategy
[hvd]: https://github.com/horovod/horovod
[hvd_tf_ex]: https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py
[nccl]: https://developer.nvidia.com/nccl
[tensorflow_containers]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags?version=23.09-tf2-py3-igpu
