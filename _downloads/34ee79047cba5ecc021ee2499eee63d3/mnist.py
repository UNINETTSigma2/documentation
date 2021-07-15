#!/usr/bin/env python

import datetime
import os
import tensorflow as tf

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
# Output to check if we are using GPU
print(tf.config.experimental.list_physical_devices('GPU'))
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
ckpt_path = os.path.join(storage_path, "checkpoints", "mnist-{epoch:04d}.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        verbose=1)

# Save initial weights
model.save_weights(ckpt_path.format(epoch=0))

# Train model with checkpointing
model.fit(x_train, y_train,
          epochs=50,
          callbacks=[ckpt_callback, tensorboard_callback],
          validation_data=(x_test, y_test),
          verbose=0)
