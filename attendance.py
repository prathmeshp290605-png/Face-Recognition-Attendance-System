import cv2
import numpy as np
import os
import streamlit as st

st.title("Face Recognition Attendance System")

# Load OpenCV model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

names = {101: "Prathmesh", 102: "Rahul"}

# Camera input through Streamlit Browser UI
img_file_buffer = st.camera_input("Take a picture for attendance")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        id_, confidence = recognizer.predict(gray[y : y + h, x : x + w])
        if confidence < 100:
            name = names.get(id_, "Unknown")
            st.success(
                f"Match Found: {name} (Confidence: {round(100 - confidence)}%)"
            )
        else:
            st.error("Face not recognized!")
