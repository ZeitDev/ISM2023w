import os
import csv
import cv2
import copy
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt

import pywt
import scipy
import torch
import skimage
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# %% 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        self.images_location = os.path.join('..', 'data', data_type)
        self.labels_location = os.path.join('..', 'data', data_type + '.csv')
        self.image_filenames = os.listdir(self.images_location)

        self.labels = {}
        with open(self.labels_location) as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                self.labels[row['name']] = row['label']

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image = skimage.io.imread(os.path.join(self.images_location, self.image_filenames[index]))
        name = os.path.splitext(self.image_filenames[index])[0]
        label = self.labels[os.path.splitext(self.image_filenames[index])[0]]
        
        return image, label, name
    
# %%
def create_dataset(data_path):
    dataset = {'names': [], 'images': [], 'labels': []}
    for image, label, name in tqdm(Dataset(data_path)):
        dataset['names'].append(name)
        dataset['images'].append(image)
        dataset['labels'].append(int(label))

        if len(dataset['images']) == 100: break

    return dataset

dataset = create_dataset(os.path.join('ISM', 'train'))

# %%
indices = {0: [], 1: [], 2: [], 3: []}
for label in indices:
    indices[label] = np.where(np.array(dataset['labels']) == label)[0]

for i in range(20):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1)
    plt.imshow(dataset['images'][indices[0][i]])

    plt.subplot(1, 4, 2)
    plt.imshow(dataset['images'][indices[1][i]])

    plt.subplot(1, 4, 3)
    plt.imshow(dataset['images'][indices[2][i]])

    plt.subplot(1, 4, 4)
    plt.imshow(dataset['images'][indices[3][i]])
    plt.show()

# %%

