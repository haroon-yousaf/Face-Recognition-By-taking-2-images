import cv2
import time

# Haar cascade classifiers (already included with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

blink_detected = False
eye_closed_frames = 0
BLINK_FRAMES_REQUIRED = 3  # Number of consecutive frames with eyes closed to register a blink

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= BLINK_FRAMES_REQUIRED:
                blink_detected = True
            eye_closed_frames = 0

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(frame, f"Blink detected: {blink_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Liveness Detection", frame)

    if blink_detected:
        print("✅ Live person detected!")
        time.sleep(1)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
