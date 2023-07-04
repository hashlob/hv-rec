import os
import numpy as np
import traceback
from PIL import Image
from imageai.Detection import ObjectDetection

# Import the necessary libraries for Streamlit deployment
import streamlit as st

# Function to perform human and vehicle recognition
def recognize_objects(image):
    try:
        # Set up object detection
        detector = ObjectDetection()
        model_path = "path/to/resnet50_coco_best_v2.1.0.h5"  # Path to the pre-trained model
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model_path)
        detector.loadModel()

        # Detect objects in the image
        detections = detector.detectObjectsFromImage(input_image=image, output_type="array")

        # Check if humans or vehicles are detected
        humans_detected = False
        vehicles_detected = False
        for detection in detections:
            if detection["name"] == "person":
                humans_detected = True
            elif detection["name"] == "car":
                vehicles_detected = True

        # Display the analyzed image with bounding boxes
        if humans_detected or vehicles_detected:
            st.image(detections["image"], channels="RGB")
            if humans_detected:
                st.success("Detected humans in the uploaded image.")
            if vehicles_detected:
                st.success("Detected vehicles in the uploaded image.")
        else:
            st.warning("No humans or vehicles detected in the uploaded image.")
    except ImportError:
        raise ImportError("imageai library is not installed. Please install it using 'pip install imageai --upgrade'.")

# Main function
def main():
    try:
        st.title('Human and Vehicle Recognition')

        # Create a file uploader in the Streamlit app
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Perform human and vehicle recognition on the image
            recognize_objects(image)

    except Exception as e:
        # Print the full error details
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
