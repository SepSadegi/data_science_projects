import tensorflow as tf
import matplotlib.pyplot as plt

def load_data():
    """Load and normalize the Fashion MNIST dataset."""
    fmnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
    # Normalize the pixel values
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    return (training_images, training_labels), (test_images, test_labels)

def build_model():
    """Build and compile the neural network model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, training_images, training_labels):
    """Train the model with the given training data."""
    history = model.fit(training_images, training_labels, epochs=5)
    return history

def visualize_convolutions(model, images, first_image, second_image, third_image, conv_number):
    """Visualize the activations of the convolutional layers."""
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    f, axarr = plt.subplots(3,4)
    for x in range(0,4):
      f1 = activation_model.predict(test_images[first_image].reshape(1, 28, 28, 1))[x]
      axarr[0,x].imshow(f1[0, : , :, conv_number], cmap='inferno')
      axarr[0,x].grid(False)
  
      f2 = activation_model.predict(test_images[second_image].reshape(1, 28, 28, 1))[x]
      axarr[1,x].imshow(f2[0, : , :, conv_number], cmap='inferno')
      axarr[1,x].grid(False)
  
      f3 = activation_model.predict(test_images[third_image].reshape(1, 28, 28, 1))[x]
      axarr[2,x].imshow(f3[0, : , :, conv_number], cmap='inferno')
      axarr[2,x].grid(False)
    plt.show()

def main():
    # Load data
    (training_images, training_labels), (test_images, test_labels) = load_data()

    # Build and compile the model
    model = build_model()

    # Train the model
    train_model(model, training_images, training_labels)

    # Evaluate the model on unseen data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize convolutions
    visualize_convolutions(model, test_images, 0, 23, 28, 1)

if __name__ == "__main__":
    main()
