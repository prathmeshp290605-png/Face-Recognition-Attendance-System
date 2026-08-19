import cv2
import csv
import os
from datetime import datetime

# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# HAAR CASCADE
# ==========================================

cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Haar Cascade file could not be loaded.")
    print("Check that this file exists:")
    print(cascade_path)
    exit()

print("Haar Cascade loaded successfully!")

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model_path = os.path.join(
    BASE_DIR,
    "trainer",
    "trainer.yml"
)

if not os.path.exists(model_path):
    print("ERROR: trainer.yml not found.")
    print("Run train.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(model_path)

print("Face recognition model loaded successfully!")

# ==========================================
# STUDENT NAMES
# ==========================================

names = {
    101: "Prathmesh",
    102: "Rahul"
}

# ==========================================
# ATTENDANCE FILE
# ==========================================

attendance_file = os.path.join(
    BASE_DIR,
    "attendance.csv"
)

# Create CSV if it doesn't exist
if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Name",
            "Date",
            "Time"
        ])

# ==========================================
# PREVENT DUPLICATE ATTENDANCE
# ==========================================

marked_today = set()

today = datetime.now().strftime("%Y-%m-%d")

if os.path.exists(attendance_file):

    with open(
        attendance_file,
        "r"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["Date"] == today:
                marked_today.add(row["ID"])

# ==========================================
# START CAMERA
# ==========================================

cam = cv2.VideoCapture(0)

if not cam.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()

print()
print("======================================")
print("      FACE ATTENDANCE SYSTEM")
print("======================================")
print("Camera started.")
print("Look at the camera.")
print("Press ESC to exit.")
print("======================================")

# ==========================================
# FACE RECOGNITION
# ==========================================

while True:

    ret, frame = cam.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break

    # Convert camera image to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # Lower confidence value = better match
        if confidence < 70:

            name = names.get(
                student_id,
                "Unknown"
            )

            display_text = f"{name}"

            # Mark attendance
            if str(student_id) not in marked_today:

                now = datetime.now()

                date = now.strftime(
                    "%Y-%m-%d"
                )

                time = now.strftime(
                    "%H:%M:%S"
                )

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        student_id,
                        name,
                        date,
                        time
                    ])

                marked_today.add(
                    str(student_id)
                )

                print(
                    f"Attendance marked: {student_id} - {name}"
                )

        else:

            name = "Unknown"
            display_text = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            display_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow(
        "Face Attendance System",
        frame
    )

    # ESC key
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

# ==========================================
# CLOSE CAMERA
# ==========================================

cam.release()

cv2.destroyAllWindows()

print()
print("Attendance system stopped.")
