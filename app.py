# 1. import required lib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# 2. preprocess - Normalize and resize the given array
def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


# 3. create random noise for given array
def random_noise(array):
    noise_factor = 0.4
    noise_array = array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)
    return np.clip(noise_array, 0.0, 1.0)


# 4. plot the random images
def plot_image(n, arr1, arr2):
    indices = np.random.randint(len(arr1), size=n)
    images1 = arr1[indices, :]
    images2 = arr2[indices, :]

    plt.figure(figsize=(20, 5))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# 5. prepare the data

num_class = 10

# load the data
(train_data, _), (test_data, _) = mnist.load_data()

# normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# add noise in the exist data
noisy_train_data = random_noise(train_data)
noisy_test_data = random_noise(test_data)

# plot the image
plot_image(num_class, train_data, noisy_train_data)

# 6. Build the autoencoder

input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder.summary()

# 7. Fit the model
autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(noisy_test_data, test_data)
)

pred = autoencoder.predict(noisy_test_data)

plot_image(5, noisy_test_data, pred)
