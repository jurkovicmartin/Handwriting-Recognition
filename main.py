import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# # 28x28 pixels digits (grayscale, white on black)
# mnist = tf.keras.datasets.mnist
# # X: pixel image data
# # Y: digit labels
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalizing (from 0-255 to 0-1)
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# # Adding layers
# # Flatten: from 2D grid to 1D vector
# # Dense: layer with neurons
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# # Output layer
# model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# # Epochs: number of times network sees the training data
# model.fit(x_train, y_train, epochs=3)

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

# model.save("handwritten_digits.keras")


model = tf.keras.models.load_model("handwritten_digits.keras")

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    # Takes only one channel from loaded image (grayscale)
    img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
    # Inverting bcs model was trained with white on black digits
    # Array bcs model expects bunch of images
    img = np.invert(np.array([img]))

    # Returns array of probabilities
    prediction = model.predict(img)
    # Index of highest probability
    max_index = np.argmax(prediction[0])

    print(f"The number is {max_index} with probability {prediction[0][max_index]}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

    image_number += 1