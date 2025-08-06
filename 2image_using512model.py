# 📌 Step 1: Import Required Libraries
import os
from deepface import DeepFace
import shutil
import tkinter as tk
from tkinter import filedialog

# 📌 Step 2: Set Folder for Copying Images
image_folder = os.path.join(os.path.expanduser("~"), "Desktop", "FaceVerificationImages")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 📌 Step 3: Function to Select Exactly 2 Images via GUI
def select_two_images_gui():
    while True:
        print("\nPlease select EXACTLY TWO face images (JPG, JPEG, PNG)...")

        root = tk.Tk()
        root.withdraw()  # Hide main window

        file_paths = filedialog.askopenfilenames(
            title="Select 2 Face Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if len(file_paths) != 2:
            print("❌ Error: You must select EXACTLY 2 images. Try again.")
        else:
            return file_paths

# 📌 Step 4: Copy Images to Target Folder
def copy_images(file_paths):
    uploaded_image_paths = []
    for path in file_paths:
        file_name = os.path.basename(path)
        target_path = os.path.join(image_folder, file_name)
        shutil.copy(path, target_path)
        uploaded_image_paths.append(target_path)
    return uploaded_image_paths

# 📌 Step 5: Compare the Two Images using Facenet512
def compare_faces_facenet512(image1, image2):
    try:
        result = DeepFace.verify(
            img1_path=image1,
            img2_path=image2,
            model_name='Facenet512',
            enforce_detection=True  # Set to False if you face detection errors
        )
        print("\n--- Verification Result ---")
        print("Result: Image 1 vs Image 2 ->", "✅ Same Person" if result["verified"] else "❌ Different People")
    except Exception as e:
        print(f"\n🚫 An error occurred during face verification: {e}")
        print("Try using high-quality, front-facing images with only one face visible.")

# 📌 Step 6: Main Function
def main():
    print("👤 Face Verification using DeepFace (Facenet512 Model)")
    file_paths = select_two_images_gui()
    
    print("\n✅ Images selected. Copying to verification folder...")
    uploaded_image_paths = copy_images(file_paths)
    
    print("✅ Copy complete.")
    print("\n🔍 Verifying faces, please wait...")

    compare_faces_facenet512(uploaded_image_paths[0], uploaded_image_paths[1])

# 📌 Step 7: Run the Code
if __name__ == "__main__":
    main()
