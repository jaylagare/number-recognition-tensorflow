import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Build a neural network model
model = Sequential([
    Input(shape=(28, 28)),  # Defines input shape (28, 28)
    Flatten(),  # Converts the 2D array into 1D array
    Dense(16, activation='sigmoid'),  # First hidden layer with 16 neurons, sigmoid activation
    Dense(16, activation='sigmoid'),  # Second hidden layer with 16 neurons, sigmoid activation
    Dense(10, activation='sigmoid')   # Output layer with 10 neurons (for 10 classes)
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='sgd', # Stochastic Gradient Descent as the optimization function 
    loss='mse',  # Mean Squared Error as the cost function
    metrics=['accuracy'])
# model.compile(optimizer='adam', # Adaptive Moment Estimation as the optimization function 
#     loss='categorical_crossentropy',  # Categorical Crossentropy as the cost function
#     metrics=['accuracy'])

# Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
# Normalize the data to a range of 0 to 1 by dividing by 255
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoding for multi-class classification
y_train = to_categorical(y_train, 10)  # One-hot encode the training labels (10 classes)
y_test = to_categorical(y_test, 10)    # One-hot encode the test labels (10 classes)

# Train the model on the training data for 5 epochs
model.fit(x_train, y_train, epochs=500)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save the trained model
try:
    model.save('model.keras')
    print("Model training complete and saved")
except Exception as e:
    print(f"Error saving model: {e}")