import os
import numpy as np
import cv2
from skimage import io

# Function Definitions
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def load_images_from_folder(folder_path):
    inp_images = []
    file_names = []
    
    # Get all files in the directory
    files = sorted(os.listdir(folder_path))
    
    for file in files:
        # Check if the file is an image (PNG or JPG)
        if file.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
            try:
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Open the image file
                image = loadImage(file_path)
                
                # Append the image and filename to our lists
                inp_images.append(image)
                file_names.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return inp_images, file_names