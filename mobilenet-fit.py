import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd

# Data augmentation transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class JamDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, num_augmentations=200):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.data_frame = pd.read_csv(csv_file)
        self.data = []
        self.labels = []

        for idx in range(len(self.data_frame)):
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
            image = Image.open(img_name).convert('RGB')
            label = 1 if self.data_frame.iloc[idx, 1] == 'Yes' else 0
            for _ in range(num_augmentations):
                if self.transform:
                    transformed_image = self.transform(image)
                else:
                    transformed_image = image
                self.data.append(transformed_image)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create a dataset instance
csv_file = "D:\\STUDY\\UNSW\\COMP9417\\work\\ass\\Data - Needs Respray - 2024-03-26\\Data - Needs Respray - 2024-03-26\\Labels-NeedsRespray-2024-03-26.csv"
root_dir = "D:\\STUDY\\UNSW\\COMP9417\\work\\ass\\Data - Needs Respray - 2024-03-26\\Data - Needs Respray - 2024-03-26\\"

dataset = JamDataset(csv_file=csv_file , root_dir=root_dir, transform=transform)

# Split the data set into training set and validation set
def split_dataset(dataset, train_split=0.8):
    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset

train_dataset, valid_dataset = split_dataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Set up model, loss function and optimizer
model = mobilenet_v3_small(pretrained=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train and validate the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}")

   # Verify model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {100 * accuracy:.2f}%")


from sklearn.metrics import classification_report
y_pred = []
y_true = []

# Test model
model.eval()
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate classification report
report = classification_report(y_true, y_pred, digits=2)
print(report)

# Calculate accuracy
accuracy = sum([pred == true for pred, true in zip(y_pred, y_true)]) / len(y_true)
print(f"Accuracy: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), 'C:\\Users\\niuka\\lab\\ass\\music\\dataset\\test\\1.pth')