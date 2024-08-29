import os
import torch
from torchvision import transforms
from PIL import Image

# Define data augmentation transformation
transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

source_directory = 'C:\\Users\\niuka\\lab\\ass\\grass\\grasstest\\test\\Yes'

target_directory = 'C:\\Users\\niuka\\lab\\ass\\grass\\grasstest\\train\\new_Yes'

#Create target folder
os.makedirs(target_directory, exist_ok=True)

# Generate 100 enhanced images from each original image
for image_name in os.listdir(source_directory):
    image_path = os.path.join(source_directory, image_name)
    image = Image.open(image_path)
    
    for i in range(100):  # Generate 100 enhanced versions of each image
        transformed_image = transform(image)
        
        # Save the enhanced image
        save_path = os.path.join(target_directory, f'{i}_{image_name}')
        transformed_image_pil = transforms.ToPILImage()(transformed_image).convert('RGB')
        transformed_image_pil.save(save_path)