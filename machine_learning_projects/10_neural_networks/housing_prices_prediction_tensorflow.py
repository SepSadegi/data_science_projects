'''
This Python code defines a simple TensorFlow model for predicting house prices based on the number of bedrooms.
'''

import numpy as np
import tensorflow as tf

def house_model():
    # Define the training data (input: number of bedrooms, output: house price in 100k)
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = 0.5 * xs + 0.5

    # Build a simple model with one Dense layer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    # Compile the model with Stochastic Gradient Descent optimizer and Mean Squared Error loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model for 1000 epochs
    model.fit(xs, ys, epochs=1000)

    return model

# Create and train the model
model = house_model()

# Make a prediction for a house with 7 bedrooms
new_x = 7.0
prediction = model.predict(np.array([new_x]))[0]
print(f'Price of the house with 7 bedrooms is {prediction} 100k')
