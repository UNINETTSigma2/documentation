```{index} GPU; TensorFlow on GPU, TensorFlow; TensorFlow on GPU
```

# TensorFlow on GPU
The intention of this guide is to teach students and researchers with access to
[Sigma2][sigma2] how to use machine learning libraries on CPU and GPU. The guide
is optimized for [`TensorFlow 2`][tensorflow], however, we hope that if you
utilize other libraries this guide still holds some value. Do not hesitate to
[contact us][contact_sigma] for additional assistance.

For the rest of this guide many of the examples ask for Sigma2 resources with
GPU. This is achieved with the `--partition=accel --gres=gpu:1`
({ref}`job_scripts_saga_accel`),
however, `TensorFlow` does not require the use of a GPU so
for testing it is recommended to not ask for GPU resources (to be scheduled
quicker) and then, once your experiments are ready to run for longer, add back
inn the request for GPU.

A complete example, both `python` and SLURM file, can be found at
[`mnist.py`](files/mnist.py) and [`run_mnist.sh`](files/run_mnist.sh).

# Installing `python` libraries
The preferred way to "install" the necessary machine learning libraries is to
load one of the pre-built [modules][module] below. By using the built-in modules
any required third-party module is automatically loaded and ready for use,
minimizing the amount of packages to load.

- `TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4`
- `PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4`
- `Theano/1.0.4-fosscuda-2019b-Python-3.7.4`
- `Keras/2.3.1-fosscuda-2019b-Python-3.7.4`

```{note}
When loading modules pressing `<tab>` gives you
autocomplete options.
```

```{note}
It can be useful to do an initial `module purge` to
ensure nothing from previous experiments is loaded before
loading modules for the first time.
```

```{note}
Modules are regularly updated so if you would like a newer
version, than what is listed above, use `module avail |
less` to browse all available packages.
```

If you need additional python libraries it is recommended to create a
[`virtualenv`][virtualenv] environment and install packages there. This
increases reproducibility and makes it easy to test different packages without
needing to install libraries globally.

Once the desired module is loaded, create a new `virtualenv` environment as
follows.

```bash
# The following will create a new folder called 'tensor_env' which will hold
# our 'virtualenv' and installed packages
$ virtualenv -p python3 tensor_env
# Next we need to activate the new environment
# NOTE: The 'Python' module loaded above must be loaded for activation to
# function, this is important when logging in and out or doing a 'module purge'
$ source tensor_env/bin/activate
# If you need to do other python related stuff outside the virtualenv you will
# need to 'deactivate' the environment with the following
$ deactivate
```

Once the environment is activated, new packages can be installed by using `pip
install <package>`. If you end up using additional packages make sure that the
`virtualenv` is activated in your [job script][sbatch_ex].

```sh
# Often useful to purge modules before running experiments
module purge

# Load machine learning library along with python
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
# Activate virtualenv
source $SLURM_SUBMIT_DIR/tensor_env/bin/activate
```

## Manual route
If you still would like to install packages through `pip` the following will
guide you through how to install the latest version of `TensorFlow` and load the
necessary packages for GPU compute.

To start, load the desired `python` version - here we will use the newest as of
writing.

```bash
$ module load Python/3.8.2-GCCcore-9.3.0
```

```{warning}
This has to be done on the login node so that we have access to
the internet and can download `pip` packages.
```

Then create a [virtual environment][virtualenv] which we will install packages
into.

```bash
# The following will create a new folder called 'tensor_env' which will hold
# our 'virtualenv' and installed packages
$ virtualenv -p python3 tensor_env
# Next we need to activate the new environment
# NOTE: The 'Python' module loaded above must be loaded for activation to
# function, this is important when logging in and out or doing a 'module purge'
$ source tensor_env/bin/activate
# If you need to do other python related stuff outside the virtualenv you will
# need to 'deactivate' the environment with the following
$ deactivate
```

Next we will install the latest version of [`TensorFlow 2`][tf_install] which
fortunately should support GPU compute directly without any other prerequisites.

```bash
$ pip install tensorflow
```

To ensure that the above install worked, start an interactive session with
`python` and run the following:

```python
>>> import tensorflow as tf
>>> tf.test.is_built_with_cuda()
# Should respond with 'True' if it worked
```

The import might show some error messages related to loading CUDA, however, for
now we just wanted to see that `TensorFlow` was installed and pre-compiled with
`CUDA` (i.e. GPU) support.

This should be it for installing necessary libraries. Below we have listed the
modules which will have to be loaded for GPU compute to work. The next code
snippet should be in your [job script][sbatch_ex] so that the correct modules
are loaded on worker nodes and the virtual environment is activated.

```sh
# Often useful to purge modules before running experiments
module purge

# Load desired modules (replace with the exact modules you used above)
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/10.1.243
module load cuDNN/7.6.4.38
# Activate virtualenv
source $SLURM_SUBMIT_DIR/tensor_env/bin/activate
```

# Loading data
For data that you intend to work on it is simplest to upload to your home area
and if the dataset is small enough simply load from there onto the worker node.

To upload data to use [`rsync`][rsync] to transfer data:

```bash
# On your own machine, upload the dataset to your home folder
$ rsync -zh --info=progress2 -r /path/to/dataset/folder <username>@saga.sigma2.no:~/.
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


## Loading built-in datasets
First we will need to download the dataset on the login node. Ensure that the
correct modules are loaded. Next open up an interactive python session with
`python`, then:

```python
>>> tf.keras.datasets.mnist.load_data()
```

This will download and cache the MNIST dataset which we can use for training
models. Load the data in your training file like so:

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
```
 
# Saving model data
For saving model data and weights we suggest the `TensorFlow` [built-in
checkpointing and save functions][tensorflow_ckpt].

The following code snippet is a more or less complete example of how to load
built-in data and save weights.

```python
#!/usr/bin/env python

# Assumed to be 'mnist.py'

import tensorflow as tf
import os

# Access storage path for '$SLURM_SUBMIT_DIR'
storage_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
			    os.environ['SLURM_JOB_ID'])

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

def create_model():
        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
                ])
        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

# Create and display summary of model
model = create_model()
# Output, such as from the following command, is outputed into the '.out' file
# produced by 'sbatch'
model.summary()

# Save model in TensorFlow format
model.save(os.path.join(storage_path, "model"))

# Create checkpointing of weights
ckpt_path = os.path.join(storage_path, "checkpoints", "mnist-{epoch:04d}.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=1)

# Save initial weights
model.save_weights(ckpt_path.format(epoch=0))

# Train model with checkpointing
model.fit(x_train[:1000], y_train[:1000],
          epochs=50,
          callbacks=[ckpt_callback],
          validation_data=(x_test[:1000], y_test[:1000]),
          verbose=0)
```

The above file can be run with the following [job script][sbatch_ex] which will
ensure that correct modules are loaded and results are copied back into your
home directory.

```sh
#!/usr/bin/bash

# Assumed to be 'mnist_test.sh'

#SBATCH --account=<your_account>
#SBATCH --job-name=<creative_job_name>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
## The following line can be omitted to run on CPU alone
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=00:30:00

# Purge modules and load tensorflow
module purge
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
# List loaded modules for reproducibility
module list

# Run python script
python $SLURM_SUBMIT_DIR/mnist.py
```

Once these two files are located on a Sigma2 resource we can run it with:

```bash
$ sbatch mnist_test.sh
```

And remember, in your code it is important to load the latest checkpoint if
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

# Using `TensorBoard`
[`TensorBoard`][tensorboard] is a nice utility for comparing different runs and
viewing progress during optimization. To enable this on Sigma2 resources we will
need to write data into our home area and some steps are necessary for
connecting and viewing the board.

We will continue to use the MNIST example from above. The following changes are
needed to enable `TensorBoard`.

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

1. Open up a terminal and connect to `Saga` as usual.
2. In the new terminal on `Saga`, load `TensorFlow` and run `tensorboard
   --logdir=/path/to/logs/fit --port=0`.
3. In the output from the above command note which port `TensorBoard` has
   started on, the last line should look something like: `TensorBoard 2.1.0 at
   http://localhost:44124/ (Press CTRL+C to quit)`.
4. Open up another terminal and this time connect to `Saga` using the following:
   `ssh -L 6006:localhost:<port> <username>@saga.sigma2.no` where `<port>` is
   the port reported from step `3` (e.g. `44124` in our case).
5. Open your browser and go to [`localhost:6006`](localhost:6006)

# Advance topics
## Using multiple GPUs
Since all of the GPU machines on `Saga` have four GPUs it can be beneficial for
some workloads to distribute the work over more than one device at a time. This
can be accomplished with the [`tf.distribute.MirroredStrategy`][mirrored].

```{warning}
As of writing, only the `MirroredStrategy` is fully
supported by `TensorFlow` which is limited to one
node at a time.
```

We will, again, continue to use the MNIST example from above. However, as we
need some larger changes to the example we will recreate the whole example and
try to highlight changes.

```python
#!/usr/bin/env python

# Assumed to be 'mnist.py'

import datetime
import os
import tensorflow as tf

# Access storage path for '$SLURM_SUBMIT_DIR'
storage_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
			    os.environ['SLURM_JOB_ID'])

## --- NEW ---
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Calculate batch size
# For your own experiments you will likely need to adjust this based on testing
# on GPUs to find the 'optimal' size
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.
## --- NEW ---
# NOTE: We need to create a 'Dataset' so that we can process the data in
# batches
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(BATCH_SIZE)

def create_model():
        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
                ])
        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

# Create and display summary of model
## --- NEW ---
with strategy.scope():
	model = create_model()
# Output, such as from the following command, is outputed into the '.out' file
# produced by 'sbatch'
model.summary()
log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
		       "logs",
		       "fit",
		       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Save model in TensorFlow format
model.save(os.path.join(storage_path, "model"))

# Create checkpointing of weights
ckpt_path = os.path.join(storage_path, "checkpoints", "mnist-{epoch:04d}.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=1)

# Save initial weights
model.save_weights(ckpt_path.format(epoch=0))

# Train model with checkpointing
model.fit(train_dataset,
          epochs=50,
	  steps_per_epoch=70,
          callbacks=[ckpt_callback, tensorboard_callback])
```

Next we will use a slightly altered job script to ask for two GPUs to see if the
above works.

```sh
#!/usr/bin/bash

# Assumed to be 'mnist_test.sh'

#SBATCH --account=<your_account>
#SBATCH --job-name=<creative_job_name>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel --gres=gpu:2
#SBATCH --time=00:30:00

# Purge modules and load tensorflow
module purge
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
# List loaded modules for reproducibility
module list

# Run python script
python $SLURM_SUBMIT_DIR/mnist.py
```

## Distributed training on multiple nodes
To utilize more than four GPUs we will turn to the [`Horovod`][hvd] project
which supports several different machine learning libraries and is capable of
utilizing `MPI`. `Horovod` is responsible for communicating between different
nodes and perform gradient computation, averaged over the different nodes.

Utilizing this library together with `TensorFlow 2` requires minimal changes,
however, there are a few things to be aware of in regards to scheduling with
`SLURM`. The following example is based on the [official `TensorFlow`
example][hvd_tf_ex].

To install `Horovod` you will need to create a `virtualenv` as described above.
Then once activated install the `Horovod` package with support for
[`NCCL`][nccl].

```sh
# This assumes that you have activated a 'virtualenv'
# $ source tensor_env/bin/activate
$ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
```

Then we can run our training using just a few modifications:

```python
#!/usr/bin/env python

# Assumed to be 'mnist_hvd.py'

import datetime
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# Initialize Horovod.
hvd.init()

# Extract number of visible GPUs in order to pin them to MPI process
gpus = tf.config.experimental.list_physical_devices('GPU')
if hvd.rank == 0:
    print(f"Found the following GPUs: '{gpus}'")
# Allow memory growth on GPU, required by Horovod
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# Since multiple GPUs might be visible to multiple ranks it is important to
# bind the rank to a given GPU
if gpus:
    print(f"Rank '{hvd.local_rank()}/{hvd.rank()}' using GPU: '{gpus[hvd.local_rank()]}'")
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
else:
    print(f"No GPU(s) configured for ({hvd.local_rank()}/{hvd.rank()})!")

# Access storage path for '$SLURM_SUBMIT_DIR'
storage_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
                            os.environ['SLURM_JOB_ID'])

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.

# Create dataset for batching
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(10000).batch(128)

# Define learning rate as a function of number of GPUs
scaled_lr = 0.001 * hvd.size()


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(scaled_lr)
    model.compile(optimizer=opt,
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model


# Create and display summary of model
model = create_model()
# Output, such as from the following command, is outputed into the '.out' file
# produced by 'sbatch'
if hvd.rank() == 0:
    model.summary()

# Create list of callback so we can separate callbacks based on rank
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other
    # processes.  This is necessary to ensure consistent initialization of all
    # workers when training is started with random weights or restored from a
    # checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to
    # worse final accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 *
    # hvd.size()` during the first three epochs. See
    # https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3,
                                             initial_lr=scaled_lr,
                                             verbose=1),
]

# Only perform the following actions on rank 0 to avoid all workers clash
if hvd.rank() == 0:
    # Tensorboard support
    log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
                           "logs",
                           "fit",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    # Save model in TensorFlow format
    model.save(os.path.join(storage_path, "model"))
    # Create checkpointing of weights
    ckpt_path = os.path.join(storage_path,
                             "checkpoints",
                             "mnist-{epoch:04d}.ckpt")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=0)
    # Save initial weights
    model.save_weights(ckpt_path.format(epoch=0))
    callbacks.extend([tensorboard_callback, ckpt_callback])

verbose = 1 if hvd.rank() == 0 else 0
# Train model with checkpointing
model.fit(x_train, y_train,
          steps_per_epoch=500 // hvd.size(),
          epochs=100,
          callbacks=callbacks,
          verbose=verbose)
```

```{warning}
When printing information it can be useful to use the `if
hvd.rank() == 0` idiom to avoid the same thing being
printed from every process.
```

This can then be scheduled with the following:

```sh
#!/usr/bin/bash

#SBATCH --account=<your project account>
#SBATCH --job-name=<fancy name>
#SBATCH --partition=accel --gres=gpu:4
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00

# Purge modules and load tensorflow
module purge
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
source $SLURM_SUBMIT_DIR/tensor_env/bin/activate
# List loaded modules for reproducibility
module list

# Export settings expected by Horovod and mpirun
export OMPI_MCA_pml="ob1"
export HOROVOD_MPI_THREADS_DISABLE=1

# Run python script
srun python $SLURM_SUBMIT_DIR/mnist_hvd.py
```

Note especially the use of `--gress=gpu:4` which means that each node allocated
for the job will get four GPUs. The `--ntasks-per-node=4` is necessary to force
`SLURM` to allocate at most four task per available node, which corresponds to
the number of GPUs per node on Saga. Also note that this means the example above
will require two nodes without using all available compute - in effect
oversubscribing. Due to this potential to oversubscribe it is recommended that
you ask for a multiple of `4` for `--ntasks` to avoid oversubscribing and
increase scheduling speed.

```{note}
**Tip:**

In the future it should be possible to use
`--gpus-per-task=1` instead to simplify and not
oversubscribe on GPUs.
```

[sigma2]: https://www.sigma2.no/
[tensorflow]: https://www.tensorflow.org/
[contact_sigma]: ../../getting_help/support_line.html
[module]: ../../software/modulescheme.html
[virtualenv]: https://virtualenv.pypa.io/en/stable/
[tf_install]: https://www.tensorflow.org/install/pip
[sbatch_ex]: ../job_scripts.html
[rsync]: https://en.wikipedia.org/wiki/Rsync
[scratch]: ../job_scripts/work_directory.html
[saga]: ../../hpc_machines/saga.html
[localscratch]: ../../files_storage/clusters.html#job-scratch-area-on-local-disk
[python_os]: https://docs.python.org/3/library/os.html
[tensorflow_ckpt]: https://www.tensorflow.org/tutorials/keras/save_and_load
[tensorboard]: https://www.tensorflow.org/tensorboard
[mirrored]: https://www.tensorflow.org/guide/distributed_training#mirroredstrategy
[hvd]: https://github.com/horovod/horovod
[hvd_tf_ex]: https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py
[nccl]: https://developer.nvidia.com/nccl
