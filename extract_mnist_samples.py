import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create a directory to save the images if it doesn't exist
if not os.path.exists('mnist_samples'):
    os.makedirs('mnist_samples')

# Loop through digits 0-9
for digit in range(10):
    # Get all indices of the current digit in the training dataset
    indices = np.where(y_test == digit)[0]
    # Randomly select one index from the found indices
    random_index = np.random.choice(indices)
    # Get the image corresponding to the digit
    image = x_test[random_index]
    
    # Save the image as a JPEG file
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.savefig(f'mnist_samples/{digit}.jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

print("Random representative images saved in the 'mnist_samples' directory.")

