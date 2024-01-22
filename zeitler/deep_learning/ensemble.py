# %%
import os
import csv
import cv2
import copy
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt

import skimage
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

data_path = os.path.join('..', 'data', 'ISM')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print('GPU detected: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('no GPU detected')

# %% Settings
isSubmission = True
NUM_CLASSES = 2

BATCH_SIZE = 20 # 16
EPOCHS = 10 # 50
LR = 1e-3
LOSS_FNC = nn.SmoothL1Loss()

SHUFFLE = True
CENTER_CROP = (512, 512)
RESIZE = (448, 448)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.15

NUM_WORKERS = 0 # WINDOWS: 0, LINUX: 6
PIN_MEMORY = False # WINDOWS: False, LINUX: True

# %%
class fully_connected(nn.Module):
    def __init__(self, model, num_ftrs):
      super(fully_connected, self).__init__()
      self.model = model
      self.fc_4 = nn.Linear(num_ftrs, 30)
      self.act_4 = nn.ReLU()
      self.dropout_4 = nn.Dropout(0.5)
      self.fc_5 = nn.Linear(30, NUM_CLASSES)

    def forward(self, x):
      x = self.model(x)
      x = torch.flatten(x, 1)
      x = self.fc_4(x)
      x = self.act_4(x)
      x = self.dropout_4(x)
      x = self.fc_5(x)
      return x

pre_model = models.densenet121(pretrained=True)
pre_model.features = nn.Sequential(pre_model.features, nn.AdaptiveAvgPool2d(output_size=(1,1)))
model = fully_connected(pre_model.features, pre_model.classifier.in_features)
model = model.to(device)
model = nn.DataParallel(model)

state_dict = torch.load(os.path.join(data_path, '..', 'KimiaNetPyTorchWeights.pth'))
for name, param in model.named_parameters():
    if name in state_dict:
        param.data.copy_(state_dict[name])
    else:
        if 'fc_5' in name:
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
        else:
            print(f"Ignoring parameter {name} as it is not in the state_dict.")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# %%
isValidation = False
class ISM_Dataset(Dataset):
    def __init__(self, data_type, isSubmission=False):
        self.isSubmission = isSubmission
        self.image_location = os.path.join(data_path, data_type)
        self.image_filenames = os.listdir(self.image_location)

        self.labels_location = os.path.join(data_path, data_type + '.csv')

        if not self.isSubmission:
          self.labels = {}
          with open(self.labels_location) as f:
              csv_reader = csv.DictReader(f)
              for row in csv_reader:
                  label = int(row['label'])
                  if label in [0, 2, 3]: label = 0
                  self.labels[row['name']] = label

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        name = os.path.splitext(self.image_filenames[index])[0]
        if not self.isSubmission:
            label = self.labels[os.path.splitext(self.image_filenames[index])[0]]
            one_hot_label = torch.zeros(1, NUM_CLASSES).squeeze(0)
            one_hot_label[label] = 1
        else:
          label = 99
          one_hot_label = 99

        image = skimage.io.imread(os.path.join(self.image_location, self.image_filenames[index]))

        if not isValidation: image = data_transforms['train'](image)
        elif isValidation: image = data_transforms['val'](image)

        return {'name': name, 'image': image, 'one_hot_label': one_hot_label, 'label': label}

def get_metrics(labels, predictions):
    accuracy = round(100*accuracy_score(labels, predictions))
    f1 = round(100*f1_score(labels, predictions, average='macro'))
    recall = round(100*recall_score(labels, predictions, average='macro'))
    precision = round(100*precision_score(labels, predictions, average='macro'))

    splitted_predictions = {}
    for label, prediction in zip(labels, predictions):
        if label not in splitted_predictions: splitted_predictions[label] = []
        splitted_predictions[label].append(prediction)

    splitted_metrics = {}
    for label in splitted_predictions:
        if label not in splitted_metrics: splitted_metrics[label] = {}
        splitted_metrics[label]['accuracy'] = round(100*accuracy_score([label]*len(splitted_predictions[label]), splitted_predictions[label]))
        splitted_metrics[label]['f1'] = round(100*f1_score([label]*len(splitted_predictions[label]), splitted_predictions[label], average='macro'))
        splitted_metrics[label]['recall'] = round(100*recall_score([label]*len(splitted_predictions[label]), splitted_predictions[label], average='macro'))
        splitted_metrics[label]['precision'] = round(100*precision_score([label]*len(splitted_predictions[label]), splitted_predictions[label], average='macro'))

    splitted_metrics = dict(sorted(splitted_metrics.items()))

    return accuracy, f1, recall, precision, splitted_metrics

# %%
data_transforms = {
    'train': transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomCrop(size=(CENTER_CROP[0], CENTER_CROP[1])),
            #transforms.Resize(size=(RESIZE[0], RESIZE[1])),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
            transforms.ToTensor(),
            #transforms.CenterCrop(size=(CENTER_CROP[0], CENTER_CROP[1])),
            #transforms.Resize(size=(RESIZE[0], RESIZE[1])),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

if 'train_dataset' not in locals():
    dataset = ISM_Dataset('train')
    dataset_lengths = [int(x * len(dataset)) for x in [TRAIN_SIZE, VAL_SIZE]]
    dataset_lengths.extend([len(dataset) - sum(dataset_lengths)])
    print(f'\nCreating datasets of sizes: train = {dataset_lengths[0]}, val = {dataset_lengths[1]}, test = {dataset_lengths[2]}')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [dataset_lengths[0], dataset_lengths[1], dataset_lengths[2]])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    dataloaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}

# %%
print('\nTraining...')
best_loss = np.inf
best_model = copy.deepcopy(model.state_dict())
best_model_epochs = []

train_loss = []
val_loss = []

start = time.time()
for epoch in range(EPOCHS):
    time_elapsed = time.time() - start
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch = {epoch+1}/{EPOCHS} | Learning Rate = {current_lr} | time = {time_elapsed // 60:.0f}:{time_elapsed % 60:.0f}')

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            isValidation = False
        elif phase == 'val':
            model.eval()
            isValidation = True

        running_loss = 0.0
        for data in dataloaders[phase]:
            images = data['image'].to(device)
            labels = data['one_hot_label'].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(images)
                loss = LOSS_FNC(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        print(f'{phase} Loss: {epoch_loss:.4f}')

        if phase == 'val' and epoch_loss < best_loss:
            print('minimum validation loss -> saving model')
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())
            best_model_epochs.append(epoch)

        if phase == 'train': train_loss.append(epoch_loss)
        if phase == 'val': val_loss.append(epoch_loss)

    scheduler.step()
    print('-' * 40)


time_elapsed = time.time() - start
print(f'Training completed in {time_elapsed // 60:.0f}:{time_elapsed % 60:.0f}s')
print(f'Best validation loss: {best_loss:.8f} at epoch {best_model_epochs[-1]+1}/{EPOCHS}')

# %%
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Epochs')

plt.legend()
plt.show()

# %%
print('Testing...')
timestamp = datetime.now().strftime("%H%M_%d%m%y")

model.load_state_dict(best_model)
bestmodel_path = os.path.join(data_path, f'bestmodel_{timestamp}.pth')
torch.save(model, bestmodel_path)

test_labels = []
test_predictions = []

with torch.no_grad():
    model.eval()
    isValidation = True
    for data in tqdm(test_loader):
        images = data['image'].to(device)
        outputs = model(images)
        top_p, top_class = F.softmax(outputs, dim=1).topk(1, dim=1)
        test_predictions.extend([x[0] for x in top_class.cpu().tolist()])

        test_labels.extend(data['label'].cpu().tolist())
    isValidation = False

accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
classification_report_result = classification_report(test_labels, test_predictions)
print(f'\nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
print()
print(classification_report_result)
print()
[print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]
print()

# %%
if isSubmission:
    submission_dataset = ISM_Dataset('test', isSubmission = True)
    submission_loader = DataLoader(submission_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print('Submission...')
    model.load_state_dict(best_model)

    submission_names = []
    submission_predictions = []

    with torch.no_grad():
        model.eval()
        isValidation = True
        for data in tqdm(submission_loader):
            images = data['image'].to(device)
            outputs = model(images)
            top_p, top_class = F.softmax(outputs, dim=1).topk(1, dim=1)

            submission_predictions.extend([x[0] for x in top_class.cpu().tolist()])
            submission_names.extend(data['name'])
        isValidation = False

    results = {'names': [], 'labels': []}
    with open(os.path.join(data_path, 'test.csv'), 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            results['names'].append(row['name'])
            index = submission_names.index(row['name'])
            results['labels'].append(submission_predictions[index])

    submission_path = os.path.join(data_path, f'submission_{timestamp}.csv')
    with open(submission_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['name', 'label'])
        for i in range(len(results['names'])):
            csv_writer.writerow([results['names'][i], results['labels'][i]])

