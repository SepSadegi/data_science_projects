"""
This script utilizes TensorFlow to construct and train a neural network model on the MNIST dataset.
MNIST consists of handwritten digits (0-9).

The goal is to train an MNIST classifier to achieve 99% accuracy and stop training when this threshold is reached.
The training should complete in less than 9 epochs.

The MNIST dataset is available directly via the tf.keras.datasets API.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load and normalize the MNIST dataset."""
    mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    (x_train, y_train), _ = mnist
    x_train = x_train / 255.0
    return x_train, y_train

def print_and_plot_image(x_train, y_train, index):
    """Print and plot a specific image from the dataset.

    Args:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        index (int): Index of the image to be printed and plotted.
    """
    np.set_printoptions(linewidth=320)
    print(f'LABEL: {y_train[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {x_train[index]}')
    plt.imshow(x_train[index], cmap='gray')
    plt.show()

class myCallback(tf.keras.callbacks.Callback):
    """Custom callback to stop training when accuracy exceeds 99%."""
    def on_epoch_end(self, epoch, logs=None):
        """
        This function is called by the callback at the end of each epoch.
        Setting up the callback to monitor the end of an epoch is a good practice,
        because with some datasets and algorithms, the loss may fluctuate within
        an epoch as not all data has been processed yet.

        Args:
          epoch (integer) - index of the epoch (required but unused in the function definition below)
          logs (dict) - metric results from the training epoch
        """
        if logs and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

def build_model():
    """Build and compile the neural network model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, callbacks):
    """Train the model with the given training data and callbacks."""
    history = model.fit(x_train, y_train, epochs=10, callbacks=callbacks)
    return history

def main():
    # Load data
    x_train, y_train = load_data()
    
    # Print dataset shape
    print(f'There are {x_train.shape[0]} examples with shape ({x_train.shape[1]},{x_train.shape[2]})')
    
    # Optionally print and plot an image
    if input("Do you want to print and plot an image? (yes/no) ").lower() == 'yes':
        index = int(input("Enter the index of the image (0-59999): "))
        print_and_plot_image(x_train, y_train, index)
    
    # Instantiate the callback
    callbacks = [myCallback()]
    
    # Build and train the model
    model = build_model()
    history = train_model(model, x_train, y_train, callbacks)

if __name__ == "__main__":
    main()
