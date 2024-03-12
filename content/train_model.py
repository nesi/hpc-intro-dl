import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

if len(sys.argv) != 2:
    raise ValueError("missing output folder parameter")

output_folder = Path(sys.argv[1])
output_folder.mkdir(parents=True, exist_ok=True)

# load training data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
input_shape = train_images.shape[1:]

# create  and compile a CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train and evaluate the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=output_folder / "logs", histogram_freq=1
)
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels),
    callbacks=[tensorboard_callback]
)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"test accuracy: {test_acc}")

# save final model
model.save(output_folder / "trained_model_cifar10")
