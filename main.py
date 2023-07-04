import os
import numpy as np
import traceback
import cv2  # Import OpenCV

# Import the necessary libraries for Streamlit deployment
import streamlit as st

# Function to perform human and vehicle recognition
def recognize_objects(image):
    try:
        # Load the pre-trained models for human and vehicle detection
        human_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        vehicle_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

        # Convert the image to grayscale for human and vehicle detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect humans in the image
        humans = human_classifier.detectMultiScale(gray_image)

        # Detect vehicles in the image
        vehicles = vehicle_classifier.detectMultiScale(gray_image)

        # Check if humans or vehicles are detected
        if len(humans) > 0 or len(vehicles) > 0:
            # Display the analyzed image with bounding boxes
            for (x, y, w, h) in humans:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (x, y, w, h) in vehicles:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the image with bounding boxes
            st.image(image, channels="BGR")

            # Print a message indicating the detection
            st.success("Detected humans or vehicles in the uploaded image.")
        else:
            st.warning("No humans or vehicles detected in the uploaded image.")
    except ImportError:
        raise ImportError("OpenCV (cv2) is not installed. Please install it using 'pip install opencv-python'.")

# Main function
def main():
    try:
        st.title('Human and Vehicle Recognition')

        # Create a file uploader in the Streamlit app
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Perform human and vehicle recognition on the image
            recognize_objects(image)

    except Exception as e:
        # Print the full error details
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
