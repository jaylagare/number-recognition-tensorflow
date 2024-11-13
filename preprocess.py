import cv2
import numpy as np
import argparse
import os

def preprocess(image_path, output_path=None):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale/"+image_path, grayscale_image) #debug
    
    # Invert colors if necessary
    average_pixel_value = np.mean(grayscale_image)
    if average_pixel_value > 128:  # Mostly bright; Adjust threshold as necessary
        grayscale_image = cv2.bitwise_not(grayscale_image)
    cv2.imwrite("inverted/"+image_path, grayscale_image) #debug

    # Normalize the image to make the highest value 255 (white)
    min_val, max_val, _, _ = cv2.minMaxLoc(grayscale_image)
    if max_val > min_val:
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val) * 255
        grayscale_image = np.uint8(grayscale_image)

    # Set lower pixel values to 0 (thresholding)
    threshold_value = 64  # Adjust this threshold as needed
    # Calculate the histogram of pixel values
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Find the most common pixel value (background) and set to 0
    threshold_value = np.argmax(hist) * 0.8  
    grayscale_image[grayscale_image < threshold_value] = 0
 
    cv2.imwrite("normalized/"+image_path, grayscale_image) #debug

    # Threshold to create a binary image (255 for content, 0 for background)
    _, binary_image = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)

    # # Find the bounding box of the non-blank regions
    coords = cv2.findNonZero(binary_image)  # Find all non-zero points (content)
    x, y, w, h = cv2.boundingRect(coords)   # Get the bounding box of content

    # # Crop the image to the bounding box
    cropped_image = grayscale_image[y:y+h, x:x+w]

    # Create a blank canvas with optimal size and border
    max_dim = max(w, h)
    border_size = int(max_dim * 0.25)  # 25% of the maximum dimension
    canvas_size = max_dim + 2 * border_size

    # Create a blank canvas with the calculated size
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Calculate the position to center the cropped image on the canvas
    y_offset = (canvas.shape[0] - h) // 2
    x_offset = (canvas.shape[1] - w) // 2

    # Place the cropped image onto the center of the canvas
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image
    
    # Resize the canvas to 28x28
    resized_image = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite("centered/"+image_path, resized_image) #debug

    # Determine output path
    if output_path is None:
        output_path = image_path  # Overwrite input image if no output path is given

    # Save the result as a JPG
    cv2.imwrite(output_path, resized_image)
    print(f"Processed image saved as {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess image for recognition.")
    parser.add_argument('input_images', nargs='+', type=str, help="Path(s) to the input image(s) (JPG format).")
    parser.add_argument('--output_image', type=str, help="Path to save the output image.")

    # Parse the arguments
    args = parser.parse_args()

    # Process input image
    for input_image in args.input_images:
        print(f"Processing {input_image}...")
        preprocess(input_image, args.output_image)
