import os
import torch
from torchvision import transforms
from PIL import Image

# 定义你的数据增强转换
transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

source_directory = 'olddataset/No'

target_directory = 'olddataset/new_No'

# 创建目标文件夹
os.makedirs(target_directory, exist_ok=True)

# 从每张原始图像生成100张增强图像
for image_name in os.listdir(source_directory):
    image_path = os.path.join(source_directory, image_name)
    image = Image.open(image_path)

    for i in range(100):  # 对于每张图片生成100个增强版本
        transformed_image = transform(image)

        # 保存增强后的图像
        save_path = os.path.join(target_directory, f'{i}_{image_name}')
        transformed_image_pil = transforms.ToPILImage()(transformed_image).convert('RGB')
        transformed_image_pil.save(save_path)