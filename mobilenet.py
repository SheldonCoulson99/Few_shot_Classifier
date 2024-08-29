import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_small
from PIL import Image
import os

#Define dataset class
class JamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['yes', 'no']
        self.data = []
        for class_label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.data.append((os.path.join(class_dir, file_name), class_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#Data preprocessing and loading
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = JamDataset(root_dir='C:\\Users\\niuka\\lab\\ass\\grass\\grasstest\\train', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = JamDataset(root_dir='C:\\Users\\niuka\\lab\\ass\\grass\\grasstest\\test', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Create model instance
model = mobilenet_v3_small(pretrained=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(torch.cuda.is_available())

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train model
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

   # Test model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {100 * accuracy:.10f}%")

    from sklearn.metrics import classification_report

y_pred = []
y_true = []

# Test model
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
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

import matplotlib.pyplot as plt

# Test the model and save the prediction results
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Get the images and labels in the test set
test_images = [img for img, _ in test_dataset]
test_labels = [label for _, label in test_dataset]

# Display images and their recognition results
plt.figure(figsize=(15, 12))
for i in range(len(test_images)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(test_images[i].permute(1, 2, 0))
    plt.title(f'Predicted: {predictions[i]}, True: {true_labels[i]}')
    plt.axis('off')

plt.tight_layout(pad=2.0)  
plt.show()
print(len(test_images))