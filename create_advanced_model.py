import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Build a neural network model
model = Sequential([
    Input(shape=(28, 28, 1)),  # Use Input to specify the shape

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')  # 10 units for 10 digit classes (0-9)
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', # Adaptive Moment Estimation as the optimization function 
    loss='categorical_crossentropy',  # Categorical Crossentropy as the cost function
    metrics=['accuracy'])

# Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
# Reshaping the training and testing data to include the channel dimension (28x28x1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalizing the data to a range of 0 to 1 by dividing by 255.0
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoding for multi-class classification
y_train = to_categorical(y_train, 10)  # One-hot encode the training labels (10 classes)
y_test = to_categorical(y_test, 10)    # One-hot encode the test labels (10 classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model on the training data for 30 epochs
model.fit(datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, lr_scheduler])

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
