import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Custom dataset for loading images and labels."""
    
    def __init__(self, image_folders, categories, max_images=200, transform=None):

        self.image_folders = image_folders  # List of image folder paths
        self.categories = categories  # Categories
        self.max_images = max_images 
        self.transform = transform 
        
        self.image_paths = []  # List of image paths
        self.labels = []  # list of labels
        
        # Populate the image paths and labels, limiting to max_images
        for folder in image_folders:
            for label_idx, category in enumerate(categories):
                category_folder = os.path.join(folder, category)
                count = 0  # Track number of images per category
                for image_file in os.listdir(category_folder):
                    if image_file.endswith((".jpg", ".jpeg", ".png")) and count < max_images:
                        image_path = os.path.join(category_folder, image_file)
                        self.image_paths.append(image_path)
                        self.labels.append(label_idx)  # Assign label (0 for COVID-19, etc.)
                        count += 1
    
    def __len__(self):
        return len(self.image_paths)  # Total number of images
    
    def __getitem__(self, idx):
        """Load and return an image and its label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read the image and apply preprocessing
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (256, 256))
        img_normalized = img_resized / 255.0
        
        # Convert to tensor
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label

def load_data(base_folder, categories=['COVID-19', 'Non-COVID', 'Normal'], max_images=200):
    """Load image paths and labels for training, validation, and testing."""
    # Folder paths
    train_folder = os.path.join(base_folder, "Train")
    val_folder = os.path.join(base_folder, "Val")
    test_folder = os.path.join(base_folder, "Test")
    
    # Dataset instances
    train_dataset = CustomDataset([train_folder], categories, max_images=max_images)
    val_dataset = CustomDataset([val_folder], categories, max_images=max_images)
    test_dataset = CustomDataset([test_folder], categories, max_images=max_images)
    
    # DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

base_folder = "/media/kunal/dual volume/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
train_loader, val_loader, test_loader = load_data(base_folder, max_images=200)
