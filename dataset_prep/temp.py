import os
import cv2
import numpy as np
from PIL import Image

def preprocess_image(input_path, output_size=(224,224)):
    """Process individual image: resize, grayscale, and save as PNG."""
    img = cv2.imread(input_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img_gray, output_size)
    
    img_normalized = img_resized / 255.0
    
    # Return the normalized image array
    return img_normalized

def load_images_from_folder(folder_path, output_size=(224, 224), max_images=150):
    """Load and preprocess images from a folder, limiting to a maximum number of images."""
    image_array = []
    count = 0
    for image_file in os.listdir(folder_path):
        if image_file.endswith((".jpg", ".jpeg", ".png")) and count < max_images:
            image_path = os.path.join(folder_path, image_file)
            processed_image = preprocess_image(image_path, output_size)
            image_array.append(processed_image)
            count += 1
    return np.array(image_array)

def load_all_data(base_folder, output_size=(224, 224), max_images=150):
    """Load images for test, train, and validate datasets from specified folder structure, limiting images per category."""
    
    # Paths for test, train, and validate sets
    test_folder = os.path.join(base_folder, "Test")
    train_folder = os.path.join(base_folder, "Train")
    validate_folder = os.path.join(base_folder, "Val")
    
    categories = ['COVID-19', 'Non-COVID', 'Normal']
    
    # Initialize lists 
    test_images = []
    train_images = []
    val_images = []
    
    # Load images for each dataset
    for dataset_folder in [test_folder, train_folder, validate_folder]:
        dataset_images = []
        
        for category in categories:
            folder_path = os.path.join(dataset_folder, category)
            images = load_images_from_folder(folder_path, output_size, max_images)
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
    # Base folder location
    base_folder = "/media/kunal/dual volume/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
    
    
    # Load data for test, train, and validation sets, limiting images to 150 per folder
    test_images, train_images, val_images = load_all_data(base_folder, max_images=150)

    temp_images= np.array([test_images, train_images, val_images])
    print(temp_images.shape)