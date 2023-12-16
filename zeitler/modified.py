'''Desc: 1:1 replication of the paper: https://univ-angers.hal.science/hal-03846675/document'''
# %%
import os
import csv
import cv2
import copy
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt

#import histomicstk as htk

import scipy
import torch
import skimage
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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

def create_dataset(data_path, test=False):
    dataset = {'names': [], 'images': [], 'labels': [], 'features': {}, 'metrics': {}}
    for image, label, name in tqdm(Dataset(data_path)):
        dataset['names'].append(name)
        dataset['images'].append(image)
        if not test: dataset['labels'].append(int(label))
        features = feature_extraction(image)
        for feature in features:
            if feature in dataset['features']: dataset['features'][feature].append(features[feature])
            else: dataset['features'][feature] = [features[feature]]
        
        if len(dataset['images']) == 500: break
        #if len(dataset['images']) == 50: break

    return dataset

def z_score_normalization(values):
    mean_value = np.mean(values)
    std_deviation = np.std(values)
    normalized_values = (values - mean_value) / std_deviation

    return normalized_values

def normalize_dataset(dataset):
    if do_z_score_normalization:
        for feature in dataset['features']:
            dataset['features'][feature] = z_score_normalization(dataset['features'][feature])

    dataset['features_combined'] = []
    for i in range(len(dataset['images'])):
        features_combined = []
        for feature in dataset['features']:
            features_combined.append(dataset['features'][feature][i])
        dataset['features_combined'].append(features_combined)

    if do_unit_normalization:
        for i in range(len(dataset['features_combined'])):
            feature_vector = np.array(dataset['features_combined'][i])
            dataset['features_combined'][i] = list(feature_vector / np.linalg.norm(feature_vector))

    return dataset

def get_vector_features(dataset):
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

        features = list(hist.flatten()) + list(normalized_edges.flatten())
        dataset['features_combined'][i].extend(list(features))

    return dataset

def get_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')

    return accuracy, f1, recall, precision

# %% Fearure Extraction
dataset = create_dataset(os.path.join('ISM', 'train'))
dataset_backup = copy.deepcopy(dataset)

# %% SETTINGS
do_z_score_normalization = True
do_unit_normalization = False

do_SVM = False
do_RF = True
do_XGB = True
do_LDA = False

do_test = False

sampling = False

# %% Postprocessing
dataset = normalize_dataset(dataset)
dataset = get_vector_features(dataset)

# %% Splitting
if not do_test:
    train_features, test_features, train_labels, test_labels = train_test_split(
        dataset['features_combined'],
        dataset['labels'],
        test_size=0.2,
        random_state=42
    )
elif do_test:
    train_features = dataset['features_combined']
    train_labels = dataset['labels']

# %% Undersampling
if sampling: train_features, train_labels = SMOTE().fit_resample(train_features, train_labels)

# %% Support Vector Machine
if do_SVM:
    from sklearn.svm import SVC
    svm_classifier = SVC(kernel='linear', C=100.0)

    svm_classifier.fit(train_features, train_labels)
    if not do_test:
        test_predictions = svm_classifier.predict(test_features)
        accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
        print('--- SVM --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Random Forest
if do_RF:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=300, criterion='gini')

    rf_classifier.fit(train_features, train_labels)
    if not do_test:
        test_predictions = rf_classifier.predict(test_features)
        accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
        print('--- RF --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Extreme Gradient Boosting
if do_XGB:
    import xgboost as xgb
    xgb_classifier = xgb.XGBClassifier(booster='gbtree', max_depth=6, eval_metric=mean_absolute_error)

    xgb_classifier.fit(train_features, train_labels)
    if not do_test:
        test_predictions = xgb_classifier.predict(test_features)
        accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
        print('--- XGBoost --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %% Linear Discriminant Analysis
if do_LDA:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_classifier = LinearDiscriminantAnalysis(solver='svd')

    lda_classifier.fit(train_features, train_labels)
    if not do_test:
        test_predictions = lda_classifier.predict(test_features)
        accuracy, f1, recall, precision = get_metrics(test_labels, test_predictions)
        print('--- LDA --- \nAccuracy: ', round(accuracy*100), '\nF1: ', round(f1*100), '\nRecall: ', round(recall*100), '\nPrecision: ', round(precision*100))

# %%
if do_test:
    test_dataset = create_dataset(os.path.join('ISM', 'test'), test=True)
    test_dataset = normalize_dataset(test_dataset)
    test_dataset = get_vector_features(test_dataset)

    true_test_features = test_dataset['features_combined']

    if do_SVM: test_predictions = svm_classifier.predict(true_test_features)
    elif do_RF: test_predictions = rf_classifier.predict(true_test_features)
    elif do_XGB: test_predictions = xgb_classifier.predict(true_test_features)
    elif do_LDA: test_predictions = lda_classifier.predict(true_test_features)

    results = {'names': [], 'labels': []}
    with open(os.path.join('..', 'data', 'ISM', 'test.csv'), 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index = test_dataset['names'].index(row['name'])
            results['names'].append(row['name'])
            results['labels'].append(test_predictions[index])

    with open(os.path.join('..', 'data', 'results', f'test_{datetime.now().strftime("%H%M_%d%m%y")}.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['name', 'label'])
        for i in range(len(results['names'])):
            csv_writer.writerow([results['names'][i], results['labels'][i]])

# %%
data = np.array(test_predictions)
integer_counts = Counter(data)
for number, count in integer_counts.items():
    print(f"{number}: {count} times")

# %%