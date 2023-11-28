'''Desc: 1:1 replication of the paper: https://univ-angers.hal.science/hal-03846675/document'''
# %%
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#import histomicstk as htk

import torch
import skimage
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# %% 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_location, labels_location, features=None, transform=None):
        self.images_location = os.path.join('data', images_location)
        self.labels_location = os.path.join('data', labels_location)
        self.features = features
        self.transform = transform

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
        if self.transform: image = self.transform(image)

        label = self.labels[os.path.splitext(self.image_filenames[index])[0]]
        if self.features: feature = self.features[index]
        else: feature = None

        return image, label, feature

def feature_extraction(image):
    image = skimage.transform.resize(image, (200, 200))
    image = skimage.filters.unsharp_mask(image, radius=2, amount=5)

    mean = np.mean(image)
    standard_deviation = np.std(image)
    median = np.median(image)
    percentile_25 = np.percentile(image, 25)
    percentile_75 = np.percentile(image, 75)

    image = skimage.color.rgb2gray(image)
    image = image.astype('uint8')
    glcm = skimage.feature.graycomatrix(image, [1], [0, np.pi/4, 2*np.pi/4, 3*np.pi/4])
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    energy = skimage.feature.graycoprops(glcm, 'energy')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')
    asm = skimage.feature.graycoprops(glcm, 'ASM')

    _, image = cv2.threshold(image*255, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    return [mean, standard_deviation, median, percentile_25, percentile_75, 
            contrast[0][0], contrast[0][1], contrast[0][2], contrast[0][3], 
            dissimilarity[0][0], dissimilarity[0][1], dissimilarity[0][2], dissimilarity[0][3], 
            homogeneity[0][0], homogeneity[0][1], homogeneity[0][2], homogeneity[0][3], 
            energy[0][0], energy[0][1], energy[0][2], energy[0][3], 
            correlation[0][0], correlation[0][1], correlation[0][2], correlation[0][3], 
            asm[0][0], asm[0][1], asm[0][2], asm[0][3], 
            hu_moments[0][0], hu_moments[1][0], hu_moments[2][0], hu_moments[3][0], 
            hu_moments[4][0], hu_moments[5][0], hu_moments[6][0]]

def get_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    return accuracy, f1, recall, precision

# %% Fearure Extraction
train_dataset = Dataset('train', 'train.csv')
features = []
for i in tqdm(range(len(train_dataset))):
    image = train_dataset[i][0]
    features.append(feature_extraction(image))
train_dataset = Dataset('train', 'train.csv', features=features)

test_dataset = Dataset('test', 'test.csv')
features = []
for i in tqdm(range(len(test_dataset))):
    image = test_dataset[i][0]
    features.append(feature_extraction(image))
test_dataset = Dataset('test', 'test_dummy.csv', features=features)

# %% Support Vector Machine
length = len(train_dataset)
new_train_dataset = []
new_test_dataset = []
for i in range(length):
    if i < length*0.8: new_train_dataset.append((train_dataset[i][0], train_dataset[i][1], train_dataset[i][2]))
    else: new_test_dataset.append((train_dataset[i][0], train_dataset[i][1], train_dataset[i][2]))

train_features = np.array([new_train_dataset[i][2] for i in range(len(new_train_dataset))])
train_labels = np.array([int(new_train_dataset[i][1]) for i in range(len(new_train_dataset))])

test_features = np.array([new_test_dataset[i][2] for i in range(len(new_test_dataset))])
test_labels = np.array([int(new_test_dataset[i][1]) for i in range(len(new_test_dataset))])

# %%
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', C=100.0)

svm_classifier.fit(train_features, train_labels)
test_predictions = svm_classifier.predict(test_features)

accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
print('--- SVM --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=300, criterion='gini')

rf_classifier.fit(train_features, train_labels)
test_predictions = rf_classifier.predict(test_features)

accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
print('--- RF --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Extreme Gradient Boosting
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(booster='gbtree', max_depth=6)

xgb_classifier.fit(train_features, train_labels)
test_predictions = xgb_classifier.predict(test_features)

accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
print('--- XGBoost --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='svd')

lda.fit(train_features, train_labels)
test_predictions = lda.predict(test_features)

accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
print('--- LDA --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), activation='relu', solver='adam', max_iter=300, batch_size=200)
mlp_classifier.out_activation_ = 'softmax'

mlp_classifier.fit(train_features, train_labels)
test_predictions = mlp_classifier.predict(test_features)

accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
print('--- MLP --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %%

