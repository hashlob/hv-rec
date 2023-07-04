import os
import numpy as np
import traceback

# Import the necessary libraries for Streamlit deployment
import streamlit as st

# Function to perform human and vehicle recognition
def recognize_objects(image_path):
    try:
        import cv2

        # Load the pre-trained models for human and vehicle detection
        human_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        vehicle_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

        # Read the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale for human and vehicle detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect humans in the image
        humans = human_classifier.detectMultiScale(gray_image)

        # Detect vehicles in the image
        vehicles = vehicle_classifier.detectMultiScale(gray_image)

        # Check if humans or vehicles are detected
        if len(humans) > 0 or len(vehicles) > 0:
            # Save the analyzed image to the ANALYZED folder
            analyzed_folder = 'C:/ANALYZED'
            analyzed_path = os.path.join(analyzed_folder, os.path.basename(image_path))
            cv2.imwrite(analyzed_path, image)

            # Print a message indicating the detection
            print(f"Detected humans or vehicles in image: {image_path}")

        # Delete the image that does not show people or cars
        os.remove(image_path)
    except ImportError:
        raise ImportError("OpenCV (cv2) is not installed. Please install it using 'pip install opencv-python'.")

# Function to process the images in the SAMPLE folder
def process_images():
    sample_folder = 'C:/SAMPLE'

    # Get the list of files in the SAMPLE folder
    image_files = os.listdir(sample_folder)

    # Iterate over each image file
    for image_file in image_files:
        # Check if the file is a JPEG image
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
            # Construct the path to the image file
            image_path = os.path.join(sample_folder, image_file)

            # Perform human and vehicle recognition on the image
            recognize_objects(image_path)

# Main function
def main():
    try:
         #Uncomment the following line for Streamlit deployment
         st.title('Human and Vehicle Recognition')

        # Process the images in the SAMPLE folder
        process_images()
    except Exception as e:
        # Print the full error details
        st.error(traceback.format_exc())

# Uncomment the following line for Streamlit deployment
 if __name__ == '__main__':
     main()
