import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
import math

# Load the trained emotion detection model
emotion_model = tf.keras.models.load_model('emotion_model.h5')
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load YOLO model for object detection
object_detector = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# Define a function to predict emotion from a face
def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))  # Resize to model input size
    face_img = face_img / 255.0  # Normalize
    face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    predictions = emotion_model.predict(face_img)
    return emotion_classes[np.argmax(predictions)]

# Calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Streamlit UI
st.title("Real-time Emotion and Object Detection with Speed Tracking")
st.text("Enable your webcam to detect objects and emotions, and track their speed in real-time.")

# OpenCV webcam capture
run = st.button("Start Webcam")
FRAME_WINDOW = st.image([])  # Placeholder for video frames

# Variables for tracking object/person positions and time
prev_person_position = None
prev_object_positions = {}
prev_time = time.time()

# Define the scale factor (meters per pixel)
# This should be determined experimentally or based on known dimensions of a reference object
SCALE_FACTOR = 0.00567  # Example scale factor (meters per pixel)

if run:
    cap = cv2.VideoCapture(0)  # Open the webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video feed. Ensure the webcam is functional.")
            break

        # Convert BGR (OpenCV format) to RGB (for Streamlit display)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects using YOLOv8
        results = object_detector(frame, stream=True)
        current_object_positions = {}
        for result in results:
            for box in result.boxes.xyxy:  # Bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box)
                label = object_detector.names[int(result.boxes.cls[0])]  # Object label
                confidence = result.boxes.conf[0].item() * 100  # Confidence score

                # Calculate speed for object
                object_center = (x1 + x2) // 2, (y1 + y2) // 2
                if label not in prev_object_positions:
                    prev_object_positions[label] = object_center
                else:
                    prev_position = prev_object_positions[label]
                    distance = calculate_distance(prev_position[0], prev_position[1], object_center[0], object_center[1])
                    current_time = time.time()
                    time_diff = current_time - prev_time  # Time between frames
                    speed_pixels_per_second = distance / time_diff  # Speed in pixels per second

                    # Convert speed to meters per second
                    speed_meters_per_second = speed_pixels_per_second * SCALE_FACTOR

                    prev_object_positions[label] = object_center  # Update position

                    # Draw object bounding box and speed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.1f}% Speed: {speed_meters_per_second:.2f} m/s", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # Detect emotions and draw bounding boxes for faces
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            emotion = predict_emotion(face)
            
            # Calculate speed for person (face)
            face_center = (x + w // 2, y + h // 2)
            if prev_person_position is None:
                prev_person_position = face_center
            else:
                prev_position = prev_person_position
                distance = calculate_distance(prev_position[0], prev_position[1], face_center[0], face_center[1])
                current_time = time.time()
                time_diff = current_time - prev_time  # Time between frames
                speed_pixels_per_second = distance / time_diff  # Speed in pixels per second

                # Convert speed to meters per second
                speed_meters_per_second = speed_pixels_per_second * SCALE_FACTOR

                prev_person_position = face_center  # Update position

                # Display emotion and speed on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {speed_meters_per_second:.2f} m/s", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prev_time = time.time()  # Update time for next iteration

    cap.release()
else:
    st.info("Click the 'Start Webcam' button to begin.")


# import cv2
# import numpy as np
# import tensorflow as tf
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image

# # Load the trained emotion detection model
# emotion_model = tf.keras.models.load_model('emotion_model.h5')
# emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Load YOLO model for object detection
# object_detector = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# # Define a function to predict emotion from a face
# def predict_emotion(face_img):
#     face_img = cv2.resize(face_img, (48, 48))  # Resize to model input size
#     face_img = face_img / 255.0  # Normalize
#     face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
#     face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
#     predictions = emotion_model.predict(face_img)
#     return emotion_classes[np.argmax(predictions)]

# # Streamlit UI
# st.title("Real-time Emotion and Object Detection")
# st.text("Enable your webcam to detect objects and emotions in real-time.")

# # OpenCV webcam capture
# run = st.button("Start Webcam")
# FRAME_WINDOW = st.image([])  # Placeholder for video frames

# if run:
#     cap = cv2.VideoCapture(0)  # Open the webcam
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Failed to capture video feed. Ensure the webcam is functional.")
#             break

#         # Convert BGR (OpenCV format) to RGB (for Streamlit display)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect objects using YOLOv8
#         results = object_detector(frame, stream=True)
#         for result in results:
#             for box in result.boxes.xyxy:  # Bounding box (x1, y1, x2, y2)
#                 x1, y1, x2, y2 = map(int, box)
#                 label = object_detector.names[int(result.boxes.cls[0])]  # Object label
#                 confidence = result.boxes.conf[0].item() * 100  # Confidence score

#                 # Draw object bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} {confidence:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

#         # Detect emotions and draw bounding boxes for faces
#         for (x, y, w, h) in faces:
#             face = gray_frame[y:y+h, x:x+w]
#             emotion = predict_emotion(face)
            
#             # Draw a rectangle and label the emotion
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Display the frame in Streamlit
#         FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     cap.release()
# else:
#     st.info("Click the 'Start Webcam' button to begin.")










# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Load the trained model
# emotion_model = tf.keras.models.load_model('emotion_model.h5')
# emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Define a function to predict emotion from a face
# def predict_emotion(face_img):
#     face_img = cv2.resize(face_img, (48, 48))  # Resize to model input size
#     face_img = face_img / 255.0  # Normalize
#     face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
#     face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
#     predictions = emotion_model.predict(face_img)
#     return emotion_classes[np.argmax(predictions)]

# # Streamlit UI
# st.title("Real-time Emotion Detection")
# st.text("Enable your webcam to detect emotions in real-time.")

# # OpenCV webcam capture
# run = st.button("Start Webcam")
# FRAME_WINDOW = st.image([])  # Placeholder for video frames

# if run:
#     cap = cv2.VideoCapture(0)  # Open the webcam
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Failed to capture video feed. Ensure the webcam is functional.")
#             break

#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

#         # Detect emotions and draw bounding boxes
#         for (x, y, w, h) in faces:
#             face = gray_frame[y:y+h, x:x+w]
#             emotion = predict_emotion(face)
            
#             # Draw a rectangle and label the emotion
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Display the frame in Streamlit
#         FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     cap.release()
# else:
#     st.info("Click the 'Start Webcam' button to begin.")

