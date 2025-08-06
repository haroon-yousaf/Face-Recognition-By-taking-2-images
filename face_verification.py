# 📌 Step 1: Ensure Libraries are Installed (You've already done this!)
# pip install deepface opencv-python tensorflow tf-keras

# 📌 Step 2: Import Required Libraries
import os
import cv2
from deepface import DeepFace
import shutil
import tkinter as tk
from tkinter import filedialog

# 📌 Step 3: Set Image Folder Path
# Create a folder on the Desktop to store the selected images
image_folder = os.path.join(os.path.expanduser("~"), "Desktop", "FaceVerificationImages")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 📌 Step 4: Open File Dialog to Select 3 Images
print("Please select EXACTLY THREE face images (e.g., JPG, PNG)")
# --- GUI Setup ---
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

file_paths = filedialog.askopenfilenames(
    title="Select 3 Face Images",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# 📌 Step 5: Check if Exactly 3 Images Were Selected
if len(file_paths) != 3:
    print("\nError: You must select EXACTLY 3 images. Please run the script again.")
    exit()

# 📌 Step 6: Copy Images to Target Folder
print("\nImages selected. Copying to verification folder...")
uploaded_image_paths = []
for path in file_paths:
    # Get just the filename (e.g., "my_photo.jpg")
    file_name = os.path.basename(path)
    # Create the full destination path
    target_path = os.path.join(image_folder, file_name)
    # Copy the file
    shutil.copy(path, target_path)
    # Add the new path to our list
    uploaded_image_paths.append(target_path)

print("Copy complete.")
print("\nVerifying faces, this may take a moment...")

# 📌 Step 7: Compare All Three Combinations
# DeepFace will download the necessary model files on the first run
try:
    result_1_2 = DeepFace.verify(img1_path=uploaded_image_paths[0], img2_path=uploaded_image_paths[1])
    result_1_3 = DeepFace.verify(img1_path=uploaded_image_paths[0], img2_path=uploaded_image_paths[2])
    result_2_3 = DeepFace.verify(img1_path=uploaded_image_paths[1], img2_path=uploaded_image_paths[2])

    # 📌 Step 8: Show Results
    print("\n--- Verification Results ---")
    print("Result: Image 1 vs Image 2 ->", "Same Person" if result_1_2["verified"] else "Different People")
    print("Result: Image 1 vs Image 3 ->", "Same Person" if result_1_3["verified"] else "Different People")
    print("Result: Image 2 vs Image 3 ->", "Same Person" if result_2_3["verified"] else "Different People")

    # 📌 Step 9: Final Conclusion
    print("\n--- Final Conclusion ---")
    if result_1_2["verified"] and result_1_3["verified"] and result_2_3["verified"]:
        print("Success: All three images appear to be of the SAME person.")
    else:
        print("Failure: The images are NOT all of the same person.")

except Exception as e:
    print(f"\nAn error occurred during face verification: {e}")
    print("This might happen if a face cannot be detected in one of the images.")