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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print('GPU detected: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('no GPU detected')
print()

# %% Settings
BATCH_SIZE = 32 # 16
EPOCHS = 40 # 50
LR = 0.01
LOSS_FNC = nn.SmoothL1Loss()

model = models.efficientnet_b0(num_classes = 4).to(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=6, gamma=0.35)

SHUFFLE = True
isTransform = True
CENTER_CROP = (350, 350)
RESIZE = (224, 224)
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2

NUM_WORKERS = 0 # WINDOWS: 0, LINUX: 6
PIN_MEMORY = False # WINDOWS: False, LINUX: True

print(f'SETTINGS:\nbatch size = {BATCH_SIZE}\nepochs = {EPOCHS}\nlearning rate = {LR}\nloss function = {LOSS_FNC}\n\nmodel = {type(model)}\noptimizer = {type(optimizer)}\nscheduler = {type(scheduler)}\n\nshuffle = {SHUFFLE}\ntransform = {isTransform}\ncenter crop = {CENTER_CROP}\nresize = {RESIZE}\ntrain size = {TRAIN_SIZE}\nval size = {VAL_SIZE}\ntest size = {int(1-TRAIN_SIZE-VAL_SIZE)}\n\nnum workers = {NUM_WORKERS}\npin memory = {PIN_MEMORY}\n')

# %% Helper Functions
class ISM_Dataset(Dataset):
    def __init__(self, data_type, isTest=False):
        self.isTest = isTest
        self.image_location = os.path.join('..', 'data', 'ISM', data_type)
        self.image_filenames = os.listdir(self.image_location)
        
        self.labels_location = os.path.join('..', 'data', 'ISM', data_type + '.csv')

        self.labels = {}
        with open(self.labels_location) as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                self.labels[row['name']] = int(row['label'])

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        name = os.path.splitext(self.image_filenames[index])[0]
        if not self.isTest: 
            label = self.labels[os.path.splitext(self.image_filenames[index])[0]]
            one_hot_label = torch.zeros(1, 4).squeeze(0)
            one_hot_label[label] = 1

        image = skimage.io.imread(os.path.join(self.image_location, self.image_filenames[index]))
        if isTransform: image = transform(image)
        
        return {'name': name, 'image': image, 'one_hot_label': one_hot_label, 'label': label}

def get_mean_std(data_type):
    print('\nCalculating mean and std...')
    pretransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(CENTER_CROP),
                transforms.Resize(RESIZE)])

    images_sum = torch.tensor([0.0])
    images_sum_squared = torch.tensor([0.0])

    image_location = os.path.join('..', 'data', 'ISM', data_type)
    image_filenames = os.listdir(image_location)
    for filename in tqdm(image_filenames):
        image = skimage.io.imread(os.path.join(image_location, filename))
        if isTransform: image = pretransform(image)

        images_sum += image.sum()
        images_sum_squared += (image ** 2).sum()

    total_pixels = RESIZE[0] * RESIZE[1] * len(image_filenames)
    mean = images_sum / total_pixels
    std = torch.sqrt((images_sum_squared / total_pixels) - (mean ** 2))

    return mean, std

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
#train_mean, train_std = get_mean_std('train')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(CENTER_CROP),
            transforms.Resize(RESIZE, antialias=True)])
            #transforms.Normalize(train_mean, train_std)])

if 'train_dataset' not in locals():
    dataset_lengths = [int(x * len(ISM_Dataset('train'))) for x in [TRAIN_SIZE, VAL_SIZE]]
    dataset_lengths.extend([len(ISM_Dataset('train')) - sum(dataset_lengths)])
    print(f'\nCreating datasets of sizes: train = {dataset_lengths[0]}, val = {dataset_lengths[1]}, test = {dataset_lengths[2]}')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ISM_Dataset('train'), [dataset_lengths[0], dataset_lengths[1], dataset_lengths[2]])
    
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
    print(f'Epoch = {epoch+1}/{EPOCHS} | Learning Rate = {scheduler.get_last_lr()[0]:.4f} | time = {time_elapsed // 60:.0f}:{time_elapsed % 60:.0f}')
    print('-' * 40)

    for phase in ['train', 'val']:
        if phase == 'train': model.train()
        elif phase == 'val': model.eval()
            
        running_loss = 0.0
        for data in dataloaders[phase]:
            images, labels = data['image'].to(device), data['one_hot_label'].to(device)

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
print(f'Best validation loss: {best_loss:.4f} at epoch {best_model_epochs[-1]}/{EPOCHS}')

# %%
print('Testing...')
model.load_state_dict(best_model)

test_labels = []
test_predictions = []

with torch.no_grad():
    model.eval()
    for data in tqdm(test_loader):
        images = data['image'].to(device)
        outputs = model(images)
        top_p, top_class = F.softmax(outputs, dim=1).topk(1, dim=1)
        test_predictions.extend([x[0] for x in top_class.cpu().tolist()])

        test_labels.extend(data['label'].cpu().tolist())

accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
print(f'\nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
[print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]
print()

# %%
