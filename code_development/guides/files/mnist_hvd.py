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
if hvd.rank() == 0:
    print(f"Found the following GPUs: '{gpus}'")

# Allow memory growth on GPU, required by Horovod
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Bind the rank to a given GPU
if gpus:
    # Each task sees only 1 GPU due to --gpus-per-task=1, so we always use index 0
    print(f"Rank '{hvd.local_rank()}/{hvd.rank()}' using GPU: '{gpus[0]}'")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
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
        tf.keras.layers.Dense(10)
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: wrap the optimizer to average gradients across all workers
    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt,
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model

# Create and display summary of model
model = create_model()

if hvd.rank() == 0:
    model.summary()

# Create list of callbacks
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3,
                                             initial_lr=scaled_lr,
                                             verbose=1 if hvd.rank() == 0 else 0),
]

# Only perform the following actions on rank 0
if hvd.rank() == 0:
    log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
                           "logs",
                           "fit",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    model.save(os.path.join(storage_path, "model"))

    ckpt_path = os.path.join(storage_path, "checkpoints", "mnist-{epoch:04d}.ckpt")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=0)

    model.save_weights(ckpt_path.format(epoch=0))
    callbacks.extend([tensorboard_callback, ckpt_callback])

# Set verbosity to 1 only for the master node
verbose = 1 if hvd.rank() == 0 else 0

# Train model using the Dataset object
model.fit(dataset,
          steps_per_epoch=500 // hvd.size(),
          epochs=100,
          callbacks=callbacks,
          verbose=verbose)
