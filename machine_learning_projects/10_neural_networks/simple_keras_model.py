'''
This project aims to use a neural network model to learn and predict the output y for a given 
input x, where the relationship is defined by the linear equation y = 2x - 1.
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Build a simple model with one Dense layer
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with Stochastic Gradient Descent optimizer and Mean Squared Error loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the training data (input and output pairs)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model for 500 epochs
model.fit(xs, ys, epochs=500)

# Make a prediction for a new input value
print(model.predict(np.array([10.0], dtype=float)))
