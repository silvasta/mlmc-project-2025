import tensorflow as tf

# from tensorflow.keras import layers, models ### import ERROR, use tensorflow.python.keras*
from tensorflow.python.keras import layers, models
import numpy as np
import visualkeras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import numpy as np
import os
import platform
import time
import random
import warnings

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="visualkeras")

seed = 142

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

### Set up environment


# Set optimizer
def get_optimizer_for_platform(learning_rate=1e-3):
    """
    Returns legacy Adam if running on Apple Silicon (M1/M2)
    to avoid known slowdowns in TF 2.11+,
    otherwise returns the standard Adam optimizer.
    """
    # Check if platform is Mac (Darwin) and Apple Silicon (arm64)
    is_apple_silicon = (platform.system() == "Darwin") and (
        "arm64" in platform.platform()
    )

    if is_apple_silicon:
        print("Detected Apple Silicon. Using legacy Adam optimizer.")
        return tf.python.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        print("Using standard Adam optimizer.")
        # TODO fix all keras import problems
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


print("TensorFlow version:", tf.__version__)

# Check for GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU detected. The following GPU(s) will be used:")
    for idx, gpu in enumerate(gpus):
        print(f"  GPU {idx}: {gpu}")
else:
    print("No GPU detected. Using CPU instead.")

### Load dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Scale images to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create a validation set from training data (e.g., 10% validation split)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, shuffle=True
)

print("Training data shape:", x_train.shape)
print("Validation data shape:", x_val.shape)
print("Test data shape:", x_test.shape)

# -----------------------------------------
# Display some images and their classes
# -----------------------------------------

# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Show the first num_images images in x_train
show_plot = False
if show_plot:
    num_images = 8
    plt.figure(figsize=(10, 2))

    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        # shift index
        shift = 33
        i = i + shift
        # Display the image
        label_index = np.argmax(y_train[i])
        plt.imshow(x_train[i])
        plt.title(class_names[label_index])
        plt.axis("off")
    plt.show()

### create the model


def build_cifar10_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Block 2
    x = layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Block 3
    x = layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    # Global Average Pooling + final dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# set up the model
model = build_cifar10_model()
model.summary()
small_model = model

# compile the model
model.compile(
    optimizer=get_optimizer_for_platform(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# set up to save model
checkpoint_path = "../models/best_model_checkpoint.weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

train_epochs = 5
train_batch_size = 64

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=train_epochs,
    batch_size=train_batch_size,
    callbacks=[checkpoint_callback],
)

# evaluate model
model.load_weights(checkpoint_path)
loss, small_model_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Best checkpoint TF model test accuracy: {small_model_acc:.4f}")
print(f"Best checkpoint TF model loss: {loss:.4f}")
# Post-Training Quantization (Full Integer)


def representative_data_gen():
    # Provide a small subset of training data for calibration
    for i in range(100):
        yield [x_train[i : i + 1]]


# Convert Keras model to TFLite with full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# Save the quantized model
quantized_model_path = "../models/quantized_model.tflite"
with open(quantized_model_path, "wb") as f:
    f.write(tflite_quant_model)

print("Quantized TFLite model saved to:", quantized_model_path)


#
#
#
print("End of file")
