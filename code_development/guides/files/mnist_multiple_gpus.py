#!/usr/bin/env python
# Assumed to be `mnist_multiple_gpus.py`

import datetime
import os
import tensorflow as tf

# Access storage path for '$SLURM_SUBMIT_DIR'
storage_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
                            os.environ['SLURM_JOB_ID'])

# --- 1. Define Distribution Strategy ---
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# --- 2. Scale Batch Size ---
# The global batch size should be multiplied by the number of GPUs
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# --- 3. Create Model and Dataset within Strategy Scope ---
with strategy.scope():
    # Load dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.

    # Pre-process dataset
    # prefetch(tf.data.AUTOTUNE) is critical for performance on HPC systems
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(60000)\
        .repeat()\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.AUTOTUNE)

    def create_model():
        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10)
                ])
        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    model = create_model()

# Display summary of model
model.summary()

# --- Logging and Checkpointing ---
log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'], "logs", "fit", 
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

# Train model with checkpointing
model.fit(train_dataset,
          epochs=50,
          steps_per_epoch=70,
          callbacks=[ckpt_callback, tensorboard_callback])