#!/usr/bin/env python

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
# Output, such as from the following command, is outputted into the '.out' file
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
