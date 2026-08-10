import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Create student's folder
student_folder = os.path.join("dataset", student_id)

if not os.path.exists(student_folder):
    os.makedirs(student_folder)

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("ERROR: Haar Cascade file could not be loaded.")
    print("Make sure haarcascade_frontalface_default.xml is in the project folder.")
    exit()


count = 0

print("Look at the camera...")
print("Press ESC to stop.")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Camera could not be opened.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        count += 1

        cv2.imwrite(
            os.path.join(student_folder, f"{count}.jpg"),
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Image {count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Register Student", frame)

    key = cv2.waitKey(100) & 0xff

    if key == 27:
        break

    if count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Face registration completed.")