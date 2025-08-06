#  Step 1: Ensure Libraries are Installed (already done)
# pip install deepface opencv-python tensorflow tf-keras

#  Step 2: Import Required Libraries
import os
import cv2
from deepface import DeepFace
import shutil
import tkinter as tk
from tkinter import filedialog

#  Step 3: Set Image Folder Path
image_folder = os.path.join(os.path.expanduser("~"), "Desktop", "FaceVerificationImages")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

#  Step 4: Function to Select Exactly 2 Images
def select_two_images():
    while True:
        print("\nPlease select EXACTLY TWO face images (JPG, JPEG, PNG)...")

        root = tk.Tk()
        root.withdraw()

        file_paths = filedialog.askopenfilenames(
            title="Select 2 Face Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if len(file_paths) != 2:
            print("❌ Error: You must select EXACTLY 2 images. Try again.")
        else:
            return file_paths

#  Step 5: Get the Valid Two Images
file_paths = select_two_images()

#  Step 6: Copy Images to Target Folder
print("\n✅ Images selected. Copying to verification folder...")
uploaded_image_paths = []
for path in file_paths:
    file_name = os.path.basename(path)
    target_path = os.path.join(image_folder, file_name)
    shutil.copy(path, target_path)
    uploaded_image_paths.append(target_path)

print("✅ Copy complete.")
print("\n🔍 Verifying faces, this may take a moment...")

#  Step 7: Compare the Two Images
try:
    result = DeepFace.verify(img1_path=uploaded_image_paths[0], img2_path=uploaded_image_paths[1])

    #  Step 8: Show Results
    print("\n--- Verification Result ---")
    print("Result: Image 1 vs Image 2 ->", "✅ Same Person" if result["verified"] else "❌ Different People")

except Exception as e:
    print(f"\n🚫 An error occurred during face verification: {e}")
    print("This might happen if a face cannot be detected in one of the images.")
