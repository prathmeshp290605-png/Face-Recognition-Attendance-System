import os
import cv2
import numpy as np
import streamlit as st

st.title("Face Recognition Attendance System")

# 1. Path Setup for trainer.yml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trainer_path = os.path.join(BASE_DIR, "trainer.yml")
cascade_path = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

# 2. Check if trainer.yml exists
if not os.path.exists(trainer_path):
    st.error(
        "❌ 'trainer.yml' file सापडली नाही! Krupaya GitHub वर trainer.yml upload kara."
    )
    st.stop()

# 3. Load Recognizer & Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Student Names Mapping
names = {101: "Prathmesh", 102: "Rahul"}

# 4. Streamlit Camera Input
st.subheader("Attendance साठी फोटो काढा")
img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
    )
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    if len(faces) == 0:
        st.warning("⚠️ चेहरा ओळखता आला नाही. व्यवस्थित समोर पाहा.")
    else:
        for x, y, w, h in faces:
            id_, confidence = recognizer.predict(gray[y : y + h, x : x + w])

            # LBPH confidence: lower is better (0 = exact match)
            if confidence < 100:
                student_name = names.get(id_, f"ID: {id_}")
                st.success(
                    f"✅ Match Found: **{student_name}** (Confidence: {round(100 - confidence)}%)"
                )
            else:
                st.error("❌ Unknown Face! (ओळख पटली नाही)")
