"""
Using a pre-trained DenseNet model for image classification to get familiar with DenseNet architecture.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Define paths
datapath = '../chest_X-ray_deep_learning'

# Load the pre-trained DenseNet121 model without the top classification layer
base_model = DenseNet121(weights=os.path.join(datapath, 'densenet.hdf5'), include_top=False)

# Print the summary of the base model to understand its architecture
print(base_model.summary())

# Print out the first five layers of the base model
print(f"First 5 layers: {base_model.layers[0:5]}")

# Print out the last five layers of the base model
print(f"Last 5 layers: {base_model.layers[-5:]}")

# Get the convolutional layers and print the first 5
conv2D_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
print(f"The first 5 Conv2D layers: {conv2D_layers[0:5]}")

# Print out the total number of convolutional layers
print(f"There are {len(conv2D_layers)} convolutional layers")

# Print the input shape of the model
print(f"The input has {base_model.input_shape[-1]} channels")
print(base_model.input)

# Print the output shape of the model
print(f"The output has {base_model.output_shape[-1]} channels")
print(base_model.output)

# Add a global spatial average pooling layer after the base model
x_pool = GlobalAveragePooling2D()(base_model.output)
print(x_pool)

# Define a set of class labels for the example
labels = ['Emphysema', 'Hernia', 'Mass', 'Pneumonia', 'Edema']
n_classes = len(labels)
print(f"In this example, we want our model to identify {n_classes} classes")

# Add a dense layer with sigmoid activation for multi-label classification
predictions = Dense(n_classes, activation="sigmoid")(x_pool)
print(f"Predictions have {n_classes} units, one for each class")
print(predictions)

# Create the final model by specifying the inputs and outputs
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model to prevent them from being trained initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print the summary of the final model
print(model.summary())