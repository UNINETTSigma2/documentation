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

# --- 3. Create Dataset (OUTSIDE the strategy scope) ---
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.

# prefetch(tf.data.AUTOTUNE) is critical for performance on HPC systems
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(60000)\
    .repeat()\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)

# Calculate correct steps to cover the whole 60k dataset
STEPS_PER_EPOCH = 60000 // BATCH_SIZE

# --- 4. Create Model within Strategy Scope ---
with strategy.scope():
    def create_model():
        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10)
                ])
        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      # FIX: Explicitly state sparse_categorical_accuracy
                      metrics=['sparse_categorical_accuracy'])
        return model

    model = create_model()

# Display summary of model
model.summary()

# --- 5. Logging and Checkpointing ---
log_dir = os.path.join(os.environ['SLURM_SUBMIT_DIR'], "logs", "fit",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create checkpointing of weights
ckpt_path = os.path.join(storage_path, "checkpoints", "mnist-{epoch:04d}.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=1)

# --- 6. Train the model ---
model.fit(train_dataset,
          epochs=50,
          steps_per_epoch=STEPS_PER_EPOCH,
          callbacks=[ckpt_callback, tensorboard_callback])

# Save the *trained* model in TensorFlow format at the very end
model.save(os.path.join(storage_path, "trained_model"))
