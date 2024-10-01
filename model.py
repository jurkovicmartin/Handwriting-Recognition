import tensorflow as tf

def load_mnist():
    """
    Loads mnist dataset and normalize it.

    Returns
    -----
    2 tuples
    (train_data, train_labels), (test_data, test_labels)
    """
    # 28x28 pixels digits (grayscale, white on black)
    mnist = tf.keras.datasets.mnist
    # X: pixel image data
    # Y: digit labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing (from 0-255 to 0-1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    return (x_train, y_train), (x_test, y_test)


def create_model(path: str, data, epochs: int):
    """
    Creates new model, trains it and saves it.

    Parameters
    -----
    path: to save the model (.keras extension)

    data: training and testing data

    epochs: number of times the model sees training data
    """
    (x_train, y_train), (x_test, y_test) = data

    model = tf.keras.models.Sequential()
    # Adding layers
    # Flatten: from 2D grid to 1D vector
    # Dense: layer with neurons
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    # Output layer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Epochs: number of times network sees the training data
    model.fit(x_train, y_train, epochs=epochs)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss is {loss}")
    print(f"Accuracy is {accuracy}")

    model.save(path)


def train_model(path: str, data, epochs: int):
    """
    Trains existing model and saves it.

    Parameters
    -----
    path: path to the model

    data: training and testing data

    epochs: number of times the model sees training data
    """
    model = tf.keras.models.load_model(path)

    (x_train, y_train), (x_test, y_test) = data

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Epochs: number of times network sees the training data
    model.fit(x_train, y_train, epochs=epochs)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss is {loss}")
    print(f"Accuracy is {accuracy}")

    model.save(path)
