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
object_detector = YOLO("yolov8n.pt")

# Define a function to predict emotion from a face
def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    predictions = emotion_model.predict(face_img)
    return emotion_classes[np.argmax(predictions)], predictions[0]

# Calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Streamlit UI
st.set_page_config(page_title="Emotion and Object Detection", page_icon=":guardsman:", layout="wide")

# Add a custom header
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #3f51b5;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            margin-top: 50px;
            color: #888;
        }
    </style>
    <div class="title">Emotion and Object Detection</div>
    <div class="subtitle">Developed by Sai Nikedh</div>
    """, unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.text("Enable your webcam to detect objects and emotions, and track their speed in real-time.")
    FRAME_WINDOW = st.image([])
    run = st.button("Start Webcam", help="Click to start detecting objects and emotions")

with col2:
    st.markdown("### Emotion Probability Distribution")
    # Create empty charts that will be updated
    emotion_chart = st.bar_chart()
    st.markdown("### Current Emotions Detected")
    emotion_text = st.empty()

# Variables for tracking object/person positions and time
prev_person_position = None
prev_object_positions = {}
prev_time = time.time()

# Define the scale factor (meters per pixel)
SCALE_FACTOR = 0.00567

if run:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video feed. Ensure the webcam is functional.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects using YOLOv8
        results = object_detector(frame, stream=True)
        current_object_positions = {}
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                label = object_detector.names[int(result.boxes.cls[0])]
                confidence = result.boxes.conf[0].item() * 100

                # Calculate and display object speed (existing code remains the same)
                object_center = (x1 + x2) // 2, (y1 + y2) // 2
                if label not in prev_object_positions:
                    prev_object_positions[label] = object_center
                else:
                    prev_position = prev_object_positions[label]
                    distance = calculate_distance(prev_position[0], prev_position[1], 
                                               object_center[0], object_center[1])
                    current_time = time.time()
                    time_diff = current_time - prev_time
                    speed_pixels_per_second = distance / time_diff
                    speed_meters_per_second = speed_pixels_per_second * SCALE_FACTOR
                    prev_object_positions[label] = object_center

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.1f}% Speed: {speed_meters_per_second:.2f} m/s",
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # Dictionary to store all detected emotions
        detected_emotions = {}

        # Detect emotions and draw bounding boxes for faces
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            emotion, probabilities = predict_emotion(face)
            
            # Update emotion probabilities for visualization
            emotion_probs = {emotion_classes[i]: float(prob) for i, prob in enumerate(probabilities)}
            
            # Update the bar chart in the sidebar
            emotion_chart.bar_chart(emotion_probs)
            
            # Store detected emotion
            detected_emotions[f"Face at ({x}, {y})"] = emotion

            # Calculate speed for person (face)
            face_center = (x + w // 2, y + h // 2)
            if prev_person_position is None:
                prev_person_position = face_center
            else:
                prev_position = prev_person_position
                distance = calculate_distance(prev_position[0], prev_position[1], 
                                           face_center[0], face_center[1])
                current_time = time.time()
                time_diff = current_time - prev_time
                speed_pixels_per_second = distance / time_diff
                speed_meters_per_second = speed_pixels_per_second * SCALE_FACTOR
                prev_person_position = face_center

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {speed_meters_per_second:.2f} m/s", 
                           (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update the emotion text display
        if detected_emotions:
            emotion_text.markdown("\n".join([f"**{k}**: {v}" for k, v in detected_emotions.items()]))
        else:
            emotion_text.markdown("No faces detected")

        # Display the frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prev_time = time.time()

    cap.release()
else:
    st.info("Click the 'Start Webcam' button to begin.")

# Footer section
st.markdown(
    """
    <div class="footer">
        <p>Emotion and Object Detection App - Powered by Streamlit, YOLO, and TensorFlow</p>
        <p>For educational purposes only. Developed by <strong>Sai Nikedh</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
