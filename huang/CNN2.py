import numpy as np
import os, cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 25

print(device)
#bat = 2

# Erstellen eines leeren DataFrame

df3 = pd.DataFrame(index=range(num_epochs), columns=['acc'])

path = "D:/Uni/Windows/TUHH/Master/1. Semester/Intelligent Systems in Medicine/Intelligent Systems in Medicine (PS)/Data set/ism_project_2023/ism_project_2023"
csv_train_path = path + "train.csv"
csv_test_path = path + "test.csv"
imgs_train_path = path + "train/"
imgs_test_path = path + "test/"

img_size = 64


CNN_linear = img_size / 8
CNN_linear = int(CNN_linear)
bn_classes = 4
test_size = 0.2
l = 0
bat = 8
bat = int(bat)
learn = 0.0001




# load image; resize
def load_and_preprocess_img(name, imgs_path):
    path = imgs_path + name + ".jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    plt.imshow(img)
    # Convert to HSV color space (if needed)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for the background color (this example assumes a blue background)
    # You need to adjust these values based on your background color
    lower_bound = np.array([110, 50, 50])
    upper_bound = np.array([130, 255, 255])

    # Thresholding the HSV image to get only the background colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Invert mask to get foreground
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to extract the foreground
    foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    plt.imshow(img)

    return img


# load csv; return dataframe with images and labels
def load_and_preprocess_data(csv_path, imgs_path=imgs_train_path):
    df = pd.read_csv(csv_path)
    df["image"] = df["name"].apply(lambda x: load_and_preprocess_img(x, imgs_path))
    return df


# load data
df = load_and_preprocess_data(csv_train_path)
# df = df.groupby('label', group_keys=False).apply(lambda x: x.iloc[125:])

# split data
if test_size > 0:
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
else:
    df_train, df_val = df, None


# display a few images with labels
# def display_images(df, n):
#     fig, ax = plt.subplots(n, n, figsize=(10, 10))
#    for i in range(n):
#        for j in range(n):
#            ax[i, j].imshow(df["image"].iloc[i * n + j])
#            ax[i, j].axis("off")
#            ax[i, j].set_title(df["label"].iloc[i * n + j])
#    plt.show()


# display_images(df_train, 3)


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx, -1]  # assuming last column is the image
        label = self.dataframe.iloc[idx, -2]  # assuming second last column is the label
        if self.transform:
            image = self.transform(image)
        return image, label


# CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        # Define the layers of the CNN
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(25, 100, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(400, 600, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(600, 800, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(800, 1000, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(1000, 1200, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(1200, 1400, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(1400, 1600, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(200 * CNN_linear * CNN_linear, 1024)  # Adjusted input features
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Define the forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        #x = self.pool(self.relu(self.conv4(x)))
       # x = self.pool(self.relu(self.conv5(x)))
       # x = self.pool(self.relu(self.conv6(x)))
       # x = self.pool(self.relu(self.conv7(x)))
       # x = self.pool(self.relu(self.conv8(x)))
        #x = self.pool(self.relu(self.conv9(x)))
        #x = self.pool(self.relu(self.conv10(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(df_train, transform=transform)
val_dataset = CustomDataset(df_val, transform=transform) if df_val is not None else None

train_loader = DataLoader(train_dataset, batch_size=bat, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bat, shuffle=False) if df_val is not None else None

# learn = 10 ** -i
# Model Initialization
model = CNN(num_classes=bn_classes)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=learn)
optimizer = optim.SGD(model.parameters(), lr=learn, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Training the Model



# Function to calculate accuracy
# Function to calculate accuracy
# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

f=0

for epoch in range(num_epochs):
    model.train()
    l = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(l)
        l = l + 1

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            val_loss += criterion(output, labels).item()

    # Lernratenanpassung basierend auf dem Validierungsverlust
    scheduler.step(val_loss)

    # Validate the model
    if val_loader:
        val_accuracy = calculate_accuracy(model, val_loader)
        #Sprint(f"Train loss: {val_loss:.6f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')
        df3.iloc[epoch, 0] = val_accuracy
        #print(df3
        #)f = f+1






print(df3)
sns.set(style="whitegrid")
plt.figure(figsize=(18, 12))
    # Use seaborn.lineplot to create line plots for each column with customized colors and line thickness
sns.lineplot(data=df3['acc'], color='darkcyan', linewidth=4, label='4', linestyle='-')
sns.scatterplot(data=df3['acc'], color='darkcyan', s=120)  # Add scatter plot for Romania


    # Set plot labels and title
plt.xlabel('Epoche', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)
plt.title('Acc', fontsize=38)

    # Adjust font size of x-axis labels (years)
plt.xticks(fontsize=20)
    # Adjust font size of x-axis labels (years)
plt.yticks(fontsize=20)

    # Adjust y-axis to start from 0
plt.ylim(0, 100)

plt.xticks(fontsize=20)
    # Adjust legend font size
plt.legend(fontsize=20)

    # Set x-axis limits to the range of years in your data
plt.xlim(df3.index.min(), df3.index.max())
# Show the plot
plt.show()