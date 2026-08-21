import torch
import os
import random
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from models.MVBCNN_18_a_new import MVBCNN, SVBCNN
# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="GVBCNN")
parser.add_argument("-BCNN_name", "--BCNN_name", type=str, help="BCNN model name", default="inception")
args = parser.parse_args()
#标定带轴

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform):
    print(f"Loading image: {image_path}")  # Debugging line
    im = Image.open(image_path).convert('RGB')
    im = im.crop((50, 50, 350, 350)).resize((1024, 1024), Image.BICUBIC)
    im = transform(im)
    return im

def load_image_x(image_path, transform):
    print(f"Loading image: {image_path}")  # Debugging line
    im = Image.open(image_path).convert('RGB')
    im = im.resize((1024, 1024), Image.BICUBIC)
    im = transform(im)
    return im

# Dataset class to handle batch processing of images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Check if the path is a valid file and not a directory
        if not os.path.isfile(image_path):
            raise ValueError(f"Path {image_path} is not a valid file.")
        
        im = load_image(image_path, self.transform)
        return im

class ImageDataset_x(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.to_pil= transforms.ToPILImage()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Check if the path is a valid file and not a directory
        if not os.path.isfile(image_path):
            raise ValueError(f"Path {image_path} is not a valid file.")
        
        im = load_image_x(image_path, self.transform)
        #im = load_image(image_path, self.transform)

        # Convert tensor to PIL image for saving
        pil_image = self.to_pil(im).convert('RGB')  # Apply the ToPILImage transform to the tensor
        pil_image = pil_image.convert('L') 
        # Create a unique file name using index, original name, or timestamp
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        save_path = os.path.join(f'{file_name}_processed{ext}')
        
        # Now save the PIL image to the specified path
        #pil_image.save(save_path)  # Save the PIL image, not the ToPILImage transform
        print(f"Saved processed image to: {save_path}")
        return im

# Function to get image features in batches
def get_image_features_batch(image_paths, model, transform, batch_size=32):
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features = []
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move data to GPU if available
            feature = model.extract_feature(inputs)  # Forward pass to get features
            feature = feature.squeeze()
            features.append(feature.cpu().numpy())  # Convert to numpy and collect features
    
    return np.vstack(features)  # Stack the features into a single 2D array

def get_image_features_x(image_paths, model, transform, batch_size=32):
    dataset = ImageDataset_x(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features = []
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move data to GPU if available
            feature = model.extract_feature(inputs,mode='task2')  # Forward pass to get features
            feature = feature.squeeze()
            features.append(feature.cpu().numpy())  # Convert to numpy and collect features
    
    return np.vstack(features)  # Stack the features into a single 2D array

# Function to compute pairwise Euclidean distances
def compute_pairwise_distances(x_features, y_features):
    # Convert numpy to torch tensors for pairwise distance calculation
    x_features_torch = torch.tensor(x_features, device=device)
    y_features_torch = torch.tensor(y_features, device=device)
    
    # Calculate pairwise Euclidean distance
    dist = torch.cdist(x_features_torch, y_features_torch, p=2)
    dist = dist.cpu().numpy()  # Move back to CPU for further processing
    return dist

# Function to process a given x image with a given set of y images in a folder
def find_top_k_closest_images(x_image_path, y_folder_path, model, transform, k=3):
    if not os.path.isfile(x_image_path):
        raise ValueError(f"Invalid image file: {x_image_path}")

    x_features = get_image_features_x([x_image_path], model, transform)
    
    y_image_files = [os.path.join(y_folder_path, f) for f in os.listdir(y_folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not y_image_files:
        raise ValueError(f"No valid images found in {y_folder_path}")

    y_features = get_image_features_batch(y_image_files, model, transform)

    cosine_similarities = np.dot(x_features, y_features.T).flatten()

    top_k_indices = np.argsort(cosine_similarities)[::-1][:k]  # 降序排列

    top_k_images = [(y_image_files[idx], float(cosine_similarities[idx])) 
                   for idx in top_k_indices]

    return top_k_images

# Main execution
if __name__ == "__main__":
    x_image_path = "path/to/x_image.png"  # Replace with your x image path
    y_folder_path = "path/to/y_images_folder"  # Replace with your y images
    
    # Load your model (SVBCNN or GVBCNN)
    model = SVBCNN("GVBCNN", pretraining=False, BCNN_name="resnet18")
    model.load_state_dict(torch.load(""))
    model.to(device)
    model.eval() 

    # Define the image transformation (normalization, etc.)
    mytransform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Find the top 3 closest images in the folder
    top_k_matches = find_top_k_closest_images(x_image_path, y_folder_path, model, mytransform, k=19)
    
    # Print the top 3 closest matches
    print(f"Top 3 closest images to {x_image_path}:")
    for idx, (image_path, distance) in enumerate(top_k_matches):
        print(f"Rank {idx + 1}: {image_path} with a distance of {distance}")
