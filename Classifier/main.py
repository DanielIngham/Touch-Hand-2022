import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import pandas as pd

"""
mnist = keras.datasets.mnist

# x - training data, y - labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(type(x_train))
print(y_train)

# normalise the data: 0,255 -> 0,1
x_train, x_test = x_train/255.0, x_test/255.0

"""

def get_dataset(filename: str) -> tuple:
    emg_data = pd.read_csv(filename)
    emg_labels = emg_data.pop("Movement")

    emg_data = np.array(emg_data)
    emg_labels = np.array(emg_labels)

    print("Labels", emg_labels)
    return emg_data, emg_labels


def train_classifier(x_train, y_train, x_test, y_test):
    # Build the model
    model = keras.models.Sequential([
        keras.layers.Dense(16),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3),
    ])

    # loss and optimiser
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # soft max
    optim = keras.optimizers.Adam(learning_rate=0.01)  # Learning Rate
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    batch_size = 10
    epochs = 10

    # Start the training
    print("Input data shape", x_train.shape, y_train.shape)
    print("TRAINING".center(80, "-"))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

    # Evaluate Model
    print("EVALUATING MODEL".center(80, "-"))
    model.evaluate(x_test, y_test, verbose=2)

    # Predictions
    print("PREDICTIONS".center(80, "-"))
    predictions = model.predict(x_test, batch_size=batch_size)
    predictions = tf.nn.softmax(predictions)

    pred0 = predictions[0]
    print("Probabilities", pred0)
    label0 = np.argmax(pred0)
    print("Prediction", label0)
    print("Answer", y_test[0])

    pred05s = predictions[0:5]
    print(pred05s.shape)
    label05s = np.argmax(pred05s, axis=1)
    print("Prediction",label05s)
    print("Answer", y_test[0:5])

if __name__ == "__main__":
    x_train, y_train = get_dataset("results/training.csv")
    x_test, y_test = get_dataset("results/testing.csv")
    train_classifier(x_train, y_train, x_test, y_test)
