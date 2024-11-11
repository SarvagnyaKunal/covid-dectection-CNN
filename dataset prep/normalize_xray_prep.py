import os
import cv2
import numpy as np
from PIL import Image

#resizing / colourconvret / formatconvert
def preprocess_image(input_path, output_size=(256, 256)):
    img = cv2.imread(input_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img_gray, output_size)
    
    base_name = os.path.splitext(input_path)[0]
    png_path = f"{base_name}.png"
    Image.fromarray(img_resized).save(png_path)

    img_array = np.array(img_resized)
    return img_array

if __name__ == "__main__":
    image_folder = "/media/kunal/dual volume/code/covid-dectection-CNN/dataset_container/working_input_data"
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, image_file)
            processed_image = preprocess_image(image_path)
