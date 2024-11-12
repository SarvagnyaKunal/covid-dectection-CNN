import os
import cv2
import numpy as np
from PIL import Image

#resizing / colourconvret / formatconvert
def preprocess_image(input_path, output_size=(256, 256)):
    img = cv2.imread(input_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img_gray, output_size)
    
    img_normalized = img_resized / 255.0
    
    # Return the normalized image array
    return img_normalized

if __name__ == "__main__":
    image_folder = "/media/kunal/dual volume/code/covid-dectection-CNN/dataset_container/working_input_data"
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, image_file)
            processed_image = preprocess_image(image_path)
    print(processed_image)