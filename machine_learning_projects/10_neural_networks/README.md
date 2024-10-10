## Neural Networks

### 1. Simple Keras Model

This project builds a simple neural network model to learn a linear relationship between input and output data.

- **Code:** simple_keras_model.py
- **Description:** The model learns to predict the output `y` for a given input `x` where `y = 2x - 1`.
- **Training Data:**
  ```python
  xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
  ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

  Prediction Example:
  print(model.predict(np.array([10.0], dtype=float)))  # Output: [[18.999987]]

### 2. House Price Prediction Model

This project builds a simple neural network model to predict house prices based on the number of bedrooms.

- **Code:** housing_prices_prediction_tensorflow.py
- **Description:** The model learns to predict the price of a house based on the number of bedrooms, following the relationship price = 0.5 * bedrooms + 0.5.
- **Training Data:**
  ```python
  xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
  ys = 0.5 * xs + 0.5

  Prediction Example:
  new_x = 7.0
  prediction = model.predict(np.array([new_x]))[0]
  print(f'Price of the house with 7 bedrooms is {prediction} 100k')  # Output: Price of the house with 7 bedrooms is 4.0000033 100k

### 3. Fashion MNIST Neural Network

This project demonstrates how to implement a neural network using TensorFlow to classify images of clothing items from the Fashion MNIST dataset.

- **Code:** fashion_mnist_neural_network.py  
- **Description:** The script constructs and trains a neural network model to classify clothing items into 10 categories from the Fashion MNIST dataset. The model is designed to stop training once the loss falls below 0.4 to ensure efficient training.  
- **Training Data:** Fashion MNIST dataset (loaded via `tf.keras.datasets.fashion_mnist`)

### Labels and Descriptions:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### 4. MNIST Neural Network

This project demonstrates how to implement a neural network using TensorFlow to classify handwritten digits from the MNIST dataset.

- **Code:** mnist_neural_network.py
- **Description:** The script constructs and trains a neural network model to classify handwritten digits from the MNIST dataset. The model is designed to achieve 99% accuracy, halting training once this threshold is reached.
- **Training Data:** MNIST dataset (loaded via `tf.keras.datasets.mnist`)
