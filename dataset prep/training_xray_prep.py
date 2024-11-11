import os
import cv2
import numpy as np
from PIL import Image

def preprocess_image(input_path, output_size=(256, 256)):
    """Process individual image: resize, grayscale, set value range b/w 0-1."""
    img = cv2.imread(input_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img_gray, output_size)
    
    img_normalized = img_resized / 255.0

    return img_normalized

def load_images_from_folder(folder_path, output_size=(256, 256),max_image=200):
    """Load and preprocess images from a folder."""
    image_array = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith((".jpg", ".jpeg", ".png"))and count < max_image:
            image_path = os.path.join(folder_path, image_file)
            processed_image = preprocess_image(image_path, output_size)
            image_array.append(processed_image)
    return np.array(image_array)

def load_all_data(base_folder, output_size=(256, 256),max_images=200):
    """Load images for test, train, and validate from dataset"""
    
    # Paths for test, train, and validate sets
    test_folder = os.path.join(base_folder, "Test")
    train_folder = os.path.join(base_folder, "Train")
    validate_folder = os.path.join(base_folder, "Val")
    
    categories = ['COVID-19', 'Non-COVID', 'Normal']
    test_images = []
    train_images = []
    val_images = []
    
    # Load images for each dataset
    for dataset_folder in [test_folder, train_folder, validate_folder]:
        dataset_images = []
        
        for category in categories:
            folder_path = os.path.join(dataset_folder, category)
            images = load_images_from_folder(folder_path, output_size=(256,256),max_images=200)
            dataset_images.append(images)
        # Append the lists for test, train, and validate datasets
        if dataset_folder == test_folder:
            test_images = dataset_images
        elif dataset_folder == train_folder:
            train_images = dataset_images
        else:
            val_images = dataset_images
    
    return test_images, train_images, val_images

if __name__ == "__main__":
    base_folder ="/media/kunal/dual volume/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
    
    # Load data for test, train, and validation sets
    test_images, train_images, val_images = load_all_data(base_folder,max_images=200)