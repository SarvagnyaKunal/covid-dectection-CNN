import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

def preprocess_image(input_path, output_size=(224, 224)):
    """Process individual image: resize, grayscale, set value range b/w 0-1."""
    img = cv2.imread(input_path)
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the required output size
    img_resized = cv2.resize(img_gray, output_size)
    
    # Normalize the image to have values between 0 and 1
    img_normalized = img_resized / 255.0

    # Return the processed image
    return img_normalized

def load_images_from_folder(folder_path, output_size=(224, 224), max_images=200):
    """Load and preprocess images from a folder."""
    image_array = []
    count = 0  # Initialize count to control max number of images
    
    for image_file in os.listdir(folder_path):
        if image_file.endswith((".jpg", ".jpeg", ".png")) and count < max_images:
            image_path = os.path.join(folder_path, image_file)
            processed_image = preprocess_image(image_path, output_size)
            image_array.append(processed_image)
            count += 1  # Increment the counter
    return np.array(image_array)

def load_all_data(base_folder, output_size=(224, 224), max_images=200):
    """Load images for test, train, and validate from dataset."""
    
    # Paths for test, train, and validate sets
    test_folder = os.path.join(base_folder, "Test")
    train_folder = os.path.join(base_folder, "Train")
    validate_folder = os.path.join(base_folder, "Val")
    
    categories = ['COVID-19', 'Non-COVID', 'Normal']

    # Initialize lists to hold the images for each dataset
    test_images = []
    train_images = []
    val_images = []
    
    # Load images for each dataset (test, train, validate)
    for dataset_folder in [test_folder, train_folder, validate_folder]:
        dataset_images = []  # Initialize list to store category images for the current dataset
        
        for category in categories:
            folder_path = os.path.join(dataset_folder, category)
            images = load_images_from_folder(folder_path, output_size=(224,224), max_images=max_images)
            dataset_images.append(images)  # Add images of the current category to dataset_images
        
        # Append the dataset images to the respectiv+
        # e lists
        if dataset_folder == test_folder:
            test_images = dataset_images
        elif dataset_folder == train_folder:
            train_images = dataset_images
        else:
            val_images = dataset_images
    
    # Convert the images into the required shape: [3, num_images, 1, 244, 244]
    # First, reshape the images to the desired format
    test_images = np.array(test_images)  # Shape: (3, num_images, 244, 244)
    train_images = np.array(train_images)  # Shape: (3, num_images, 256, 256)
    val_images = np.array(val_images)  # Shape: (3, num_images, 256, 256)

    '''Now reshape the images by adding an extra dimension for channels (grayscale)'''
    # test_images = test_images.reshape(3, max_images, 1, 256, 256)
    # train_images = train_images.reshape(3, max_images, 1, 256, 256)
    # val_images = val_images.reshape(3, max_images, 1, 256, 256)

    return test_images, train_images, val_images

if __name__ == "__main__":
    base_folder = "c:/Users/sarva/Desktop/nf/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
    
    # Load data for test, train, and validation sets
    test_images, train_images, val_images = load_all_data(base_folder, max_images=200)
    
    # Print the shape to verify
    print(f"Test Images Shape: {test_images.shape}")
    # test_image = test_images[0,1,0]
    # plt.imshow(test_image) 
    # plt.show()