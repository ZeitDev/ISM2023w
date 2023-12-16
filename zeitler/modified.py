'''Desc: 1:1 replication of the paper: https://univ-angers.hal.science/hal-03846675/document'''
# %%
import os
import csv
import cv2
import copy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#import histomicstk as htk

import scipy
import torch
import skimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# %% 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type = os.path.join('ISM', 'train')):
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
        label = self.labels[os.path.splitext(self.image_filenames[index])[0]]

        return image, label

def feature_extraction(image):
    image_rgb = (skimage.filters.unsharp_mask(image, radius=3, amount=7) * 255).astype('uint8')
    image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')

    mean = np.mean(image_rgb)
    standard_deviation = np.std(image_rgb)
    median = np.median(image_rgb)
    percentile_25 = np.percentile(image_rgb, 25)
    percentile_75 = np.percentile(image_rgb, 75)
    kurtosis = scipy.stats.kurtosis(image_rgb.flatten())

    glcm = skimage.feature.graycomatrix(image_gray, [1], [0, np.pi/4, 2*np.pi/4, 3*np.pi/4])
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    energy = skimage.feature.graycoprops(glcm, 'energy')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')
    asm = skimage.feature.graycoprops(glcm, 'ASM')

    _, image_thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(image_thresh)
    hu_moments = cv2.HuMoments(moments)

    return {'mean': mean, 'standard_deviation': standard_deviation, 'median': median, 'percentile_25': percentile_25, 'percentile_75': percentile_75, 'kurtosis': kurtosis, 
            'contrast_0': contrast[0][0], 'contrast_1': contrast[0][1], 'contrast_2': contrast[0][2], 'contrast_3': contrast[0][3],
            'dissimilarity_0': dissimilarity[0][0], 'dissimilarity_1': dissimilarity[0][1], 'dissimilarity_2': dissimilarity[0][2], 'dissimilarity_3': dissimilarity[0][3],
            'homogeneity_0': homogeneity[0][0], 'homogeneity_1': homogeneity[0][1], 'homogeneity_2': homogeneity[0][2], 'homogeneity_3': homogeneity[0][3],
            'energy_0': energy[0][0], 'energy_1': energy[0][1], 'energy_2': energy[0][2], 'energy_3': energy[0][3],
            'correlation_0': correlation[0][0], 'correlation_1': correlation[0][1], 'correlation_2': correlation[0][2], 'correlation_3': correlation[0][3],
            'asm_0': asm[0][0], 'asm_1': asm[0][1], 'asm_2': asm[0][2], 'asm_3': asm[0][3],
            'hu_moments_0': hu_moments[0][0], 'hu_moments_1': hu_moments[1][0], 'hu_moments_2': hu_moments[2][0], 'hu_moments_3': hu_moments[3][0], 'hu_moments_4': hu_moments[4][0], 'hu_moments_5': hu_moments[5][0], 'hu_moments_6': hu_moments[6][0],
            }

def z_score_normalization(values):
    mean_value = np.mean(values)
    std_deviation = np.std(values)
    normalized_values = (values - mean_value) / std_deviation

    return normalized_values

def get_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')

    return accuracy, f1, recall, precision

# %% Fearure Extraction
data_path = os.path.join('ISM', 'train')
dataset = {'images': [], 'labels': [], 'features': {}}
for image, label in tqdm(Dataset(data_path)):
    dataset['images'].append(image)
    dataset['labels'].append(int(label))
    features = feature_extraction(image)
    for feature in features:
        if feature in dataset['features']: dataset['features'][feature].append(features[feature])
        else: dataset['features'][feature] = [features[feature]]

    #if len(dataset['labels']) == 500: break
    #if len(dataset['labels']) == 1: break

dataset_backup = copy.deepcopy(dataset)

# %% SETTINGS
do_z_score_normalization = True
do_unit_normalization = False

# %% Normalization
if do_z_score_normalization:
    for feature in dataset['features']:
        dataset['features'][feature] = z_score_normalization(dataset['features'][feature])

dataset['features_combined'] = []
for i in range(len(dataset['labels'])):
    features_combined = []
    for feature in dataset['features']:
        features_combined.append(dataset['features'][feature][i])
    dataset['features_combined'].append(features_combined)

if do_unit_normalization:
    for i in range(len(dataset['features_combined'])):
        feature_vector = np.array(dataset['features_combined'][i])
        dataset['features_combined'][i] = list(feature_vector / np.linalg.norm(feature_vector))

# %% Vector Feature Extraction
for i in tqdm(range(len(dataset['images']))):
    image = dataset['images'][i]
    image_rgb = (skimage.filters.unsharp_mask(image, radius=3, amount=7) * 255).astype('uint8')
    image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')
    
    # Color Histogram
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = hist / hist.sum()

    # Edge Detection
    edges = cv2.Canny(image_gray, 50, 150)
    normalized_edges = edges / 255.0

    # Scale Invariant Feature Transform
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)



    features = list(hist.flatten()) + list(normalized_edges.flatten())
    dataset['features_combined'][i].extend(list(features))



# %% Splitting
train_features, test_features, train_labels, test_labels = train_test_split(
    dataset['features_combined'],
    dataset['labels'],
    test_size=0.2,
    random_state=42
)

# %% Support Vector Machine
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

# %%

