import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse

# Load the pre-trained model
model = tf.keras.models.load_model('model.keras')

# Function to preprocess and predict the digit from images
def recognize_digit(image_path):
    # Load the image
    img = image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))

    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability
    return predicted_class

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Recognize character in 28x28 image")
    parser.add_argument('input_images', nargs='+', type=str, help="Path(s) to the input color image(s) (JPG format).")

    # Parse the arguments
    args = parser.parse_args()

    # Process each input image
    for input_image in args.input_images:
        print(f'Image: {input_image}')
        predicted_digit = recognize_digit(input_image)
        print(f'Recognized Digit: {predicted_digit}')
