import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

def select_images():
    root = tk.Tk()
    root.withdraw()
    while True:
        file_paths = filedialog.askopenfilenames(
            title="Select exactly 2 face images",
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if len(file_paths) != 2:
            messagebox.showerror("Error", "Please select exactly 2 images.")
        else:
            return file_paths

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Unable to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: Histogram equalization to improve lighting conditions
    gray = cv2.equalizeHist(gray)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ No face found in: {image_path}")
        return None

    # Select the largest face (most likely to be the actual face)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]
    face_crop = gray[y:y+h, x:x+w]

    # Resize to 256x256 with better interpolation
    face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    face_crop = face_crop.astype(np.float32) / 255.0

    return face_crop

def compare_faces(face1, face2):
    try:
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((face1 - face2) ** 2)
        similarity = 100 - (mse * 100)  # similarity in percentage

        print(f"\n🔍 MSE: {mse:.6f}")
        print(f"✅ Similarity: {similarity:.2f}%")

        if similarity > 90:
            print("🟢 Faces are highly similar (possibly same person).")
        elif similarity > 70:
            print("🟡 Faces are moderately similar.")
        else:
            print("🔴 Faces are different.")
    except Exception as e:
        print("Error comparing faces:", e)

def main():
    try:
        img_paths = select_images()
        face1 = detect_and_crop_face(img_paths[0])
        face2 = detect_and_crop_face(img_paths[1])

        if face1 is not None and face2 is not None:
            compare_faces(face1, face2)
        else:
            print("❌ Could not detect faces in one or both images.")
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

if __name__ == "__main__":
    main()
