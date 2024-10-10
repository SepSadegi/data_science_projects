"""
This script utilizes TensorFlow to construct and train a neural network model on the Fashion MNIST dataset.
The dataset comprises 10 different types of clothing items, and the neural network is designed to recognize
and classify these items accurately.

Label    Description
0        T-shirt/top
1        Trouser
2        Pullover
3        Dress
4        Coat
5        Sandal
6        Shirt
7        Sneaker
8        Bag
9        Ankle boot

The dataset is available directly in the tf.keras.datasets API.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data():
    """Load and normalize the Fashion MNIST dataset."""
    fmnist = tf.keras.datasets.fashion_mnist
    # load_data() returns two tuples, each containing two lists: training and test split
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
    '''
    *** Neural Network works better with normalized data ***
    Ensure better learning of the neural network, particularly in image processing,
    by scaling all values between 0 and 1. This process is known as normalization.
    If we don't normalize the data, following issues can arise:
    1. Slower Training
    2. Poor Performance: larger loss
    3. Unstable Training: exploding or vanishing gradients
    4. Difficulty in learning Patterns
    '''
    training_images = training_images / 255.0 # because each image is a grid of value from 0 to 255 with pixel grayscale values.
    test_images = test_images / 255.0
    return (training_images, training_labels), (test_images, test_labels)

def print_and_plot_image(training_images, training_labels, index):
    """Print and plot a specific image from the dataset.

    Args:
        training_images (numpy.ndarray): Training images.
        training_labels (numpy.ndarray): Training labels.
        index (int): Index of the image to be printed and plotted. (between 0 to 59999)
    """
    # Set number of characters per row when printing
    np.set_printoptions(linewidth=320)
    print(f'LABEL: {training_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')
    plt.imshow(training_images[index], cmap='gray')
    plt.show()

class myCallback(tf.keras.callbacks.Callback):
    """Custom callback to stop training when loss falls below 0.4."""
    def on_epoch_end(self, epoch, logs=None): # logs={}
        """
        This function is called by the callback at the end of each epoch.
        Setting up the callback to monitor the end of an epoch is a good practice,
        because with some datasets and algorithms, the loss may fluctuate within
        an epoch as not all data has been processed yet.

        Args:
          epoch (integer) - index of the epoch (required but unused in the function definition below)
          logs (dict) - metric results from the training epoch
        """
        # Can be performed on accuracy or loss
        # accuracy: determine if the model is performing well. logs.get('accuracy') >= 0.6:
        # loss: measures how well the model's predictions match the label (minimizing error). Lower --> better
        # Check the loss
        if logs and logs.get('loss') < 0.4:
            print("\nLoss is lower than 0.4 so cancelling the training!")
            # print("\nAccuracy is higher than 60% so cancelling training!")
            self.model.stop_training = True # Stop if the threshold is met

def build_model():
    """Build and compile the neural network model."""
    # Build the classification model (There is an input layer in the shape of the data
    #                                 an output layer in the shape of the classes,
    #                                 and one hidden layer that tries to figure out the rules between them.)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # 784x1
        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Increasing neurons from 128 to 1024 increases training time and accuracy
        tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Number of classes
    ])
    model.compile(optimizer='adam', # tf.optimizers.Adam()
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, training_images, training_labels, callbacks):
    """Train the model with the given training data and callbacks."""
    # history = model.fit(training_images, training_labels, epochs=15) # More epochs --> better loss (Be careful about overfitting! e.g. epochs=30)
    history = model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])
    return history

def main():
    # Load data
    (training_images, training_labels), (test_images, test_labels) = load_data()

    # Optionally print and plot an image
    if input("Do you want to print and plot an image? (yes/no) ").lower() == 'yes':
        index = int(input("Enter the index of the image (0-59999): "))
        print_and_plot_image(training_images, training_labels, index)

    # Instantiate the callback
    callbacks = [myCallback()]

    # Build and compile the model
    model = build_model()

    # Explanation or demonstration of the activation function and prediction process
    # ===== *** For learning and understanding purpose *** =====
    # Declare sample input and convert to a tensor
    inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
    inputs = tf.convert_to_tensor(
        inputs)  # to make sure that the data is represented in the format expected by TensorFlow
    print(f'input to softmax function: {inputs.numpy()}')

    # Feed the inputs to a softmax activation function
    outputs = tf.keras.activations.softmax(inputs)
    print(f'output of softmax function: {outputs.numpy()}')

    # Get the sum of all values after the softmax
    sum_outputs = tf.reduce_sum(outputs)
    print(f'Sum of outputs: {sum_outputs}')

    # Get the index with the highest value
    prediction = np.argmax(outputs)
    print(f'Class with the highest probability: {prediction}')
    # ===== *** ===================================== *** =====

    # Train the model
    train_model(model, training_images, training_labels, callbacks)

    # Evaluate the model on unseen data
    model.evaluate(test_images, test_labels)

    # Exercise: Create classifications for each of the test images
    classifications = model.predict(test_images)

    # Print the first entry in the classifications
    # Explanation: The output of the model is a list of 10 numbers.
    # Each value in the list corresponds to the predicted probability for each class (0-9).
    # *** It's probability that this item is each of the 10 classes ***
    print(classifications[0])
    print(test_labels[0])

if __name__ == "__main__":
    main()
