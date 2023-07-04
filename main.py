import os
import numpy as np
import traceback
from PIL import Image
import cv2

# Import the necessary libraries for Streamlit deployment
import streamlit as st

# Function to perform human and vehicle recognition
def recognize_objects(image):
    try:
        # Load the pre-trained YOLO model
        net = cv2.dnn.readNetFromDarknet("path/to/yolov3.cfg", "path/to/yolov3.weights")

        # Define the classes for detection (person and car)
        classes = ["person", "car"]

        # Extract the dimensions of the image
        (h, w) = image.shape[:2]

        # Create a blob from the image and perform forward pass through the network
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        # Initialize lists for bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process each output layer
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in [0, 2]:  # Filter detections for person and car
                    # Scale the bounding box coordinates to the original image size
                    box = detection[0:4] * np.array([w, h, w, h])
                    (center_x, center_y, width, height) = box.astype("int")

                    # Calculate the top-left corner coordinates of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # Add the bounding box coordinates, confidences, and class IDs to the respective lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Check if humans or vehicles are detected
        humans_detected = False
        vehicles_detected = False
        for i in indices:
            idx = i[0]
            if class_ids[idx] == 0:
                humans_detected = True
            elif class_ids[idx] == 2:
                vehicles_detected = True

            # Draw bounding boxes on the image
            x, y, w, h = boxes[idx]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, classes[class_ids[idx]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the analyzed image with bounding boxes
        if humans_detected or vehicles_detected:
            st.image(image, channels="BGR")
            if humans_detected:
                st.success("Detected humans in the uploaded image.")
            if vehicles_detected:
                st.success("Detected vehicles in the uploaded image.")
        else:
            st.warning("No humans or vehicles detected in the uploaded image.")
    except ImportError:
        raise ImportError("OpenCV is not installed. Please install it using 'pip install opencv-python'.")

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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Perform human and vehicle recognition on the image
            recognize_objects(image)

    except Exception as e:
        # Print the full error details
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
