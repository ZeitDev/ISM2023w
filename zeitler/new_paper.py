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

import pywt
import scipy
import torch
import skimage
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


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
    image_sharpened = skimage.filters.unsharp_mask(image, radius=0.5, amount=2)

    image_rgb = image_sharpened #(image_sharpened * 255).astype('uint8')
    #image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')

    image_red = image_rgb[:, :, 0]
    image_green = image_rgb[:, :, 1]
    image_blue = image_rgb[:, :, 2]

    # 2D-DFT
    dft_shift_red = np.fft.fftshift(np.fft.fft2(image_red))
    dft_shift_green = np.fft.fftshift(np.fft.fft2(image_green))

    magnitude_spectrum_red = 20 * np.log(np.abs(dft_shift_red))
    magnitude_spectrum_green = 20 * np.log(np.abs(dft_shift_green))

    magnitude_spectrum_red = magnitude_spectrum_red / np.max(magnitude_spectrum_red)
    magnitude_spectrum_green = magnitude_spectrum_green / np.max(magnitude_spectrum_green)

    dft_red = skimage.transform.resize(magnitude_spectrum_red, (64, 64))
    dft_green = skimage.transform.resize(magnitude_spectrum_green, (64, 64))

    # 2D-DWT
    _, (cH, _, _) = pywt.dwt2(image_rgb, 'haar')
    cH_red = skimage.transform.resize(cH[:, :, 0], (64, 64))
    cH_green = skimage.transform.resize(cH[:, :, 1], (64, 64))

    # feature normalization
    # dft_red = z_score_normalization(dft_red.flatten())
    # dft_green = z_score_normalization(dft_green.flatten())
    # cH_red = z_score_normalization(cH_red.flatten())
    # cH_green = z_score_normalization(cH_green.flatten())

    # features_combined = np.concatenate([dft_red, dft_green, cH_red, cH_green])

    return {'dft_red': dft_red.flatten(), 'dft_green': dft_green.flatten(), 'cH_red': cH_red.flatten(), 'cH_green': cH_green.flatten()}

def create_dataset(data_path, test=False):
    dataset = {'names': [], 'labels': [], 'features': {}, 'metrics': {}}
    for image, label, name in tqdm(Dataset(data_path)):
        dataset['names'].append(name)
        if not test: dataset['labels'].append(int(label))
        features = feature_extraction(image)
        for feature in features:
            if feature in dataset['features']: np.append(dataset['features'][feature], features[feature])
            else: dataset['features'][feature] = features[feature]
        
        if len(dataset['names']) == 500: break
        #if len(dataset['names']) == 250: break
        #if len(dataset['names']) == 20: break

    return dataset

def normalize_dataset(dataset):
    for feature in dataset['features']:
        dataset['features'][feature] = z_score_normalization(dataset['features'][feature].flatten())

    for i in range(len(dataset['names'])):
        features_combined = []
        for feature in dataset['features']:
            feature_list = list(dataset['features'][feature])
            features_combined.append(feature_list[i])
        dataset['features_combined'].append(features_combined)

    dataset['features_combined'] = np.concatenate([dataset['features']['dft_red'], dataset['features']['dft_green'], dataset['features']['cH_red'], dataset['features']['cH_green']])

    return dataset

def z_score_normalization(values):
    mean_value = np.mean(values)
    std_deviation = np.std(values)
    normalized_values = (values - mean_value) / std_deviation

    return normalized_values

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

# %% SETTINGS
do_SVM = True
do_RF = True
do_XGB = True
do_LDA = True

do_pca = False
do_ensemble = False

do_test = False

sampling = False

# %% Datatset
if 'dataset' not in locals():
    dataset = create_dataset(os.path.join('ISM', 'train'))
    dataset = normalize_dataset(dataset)
    dataset['labels'] = np.array(dataset['labels'])

    if do_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=500)
        dataset['features_combined'] = pca.fit_transform(dataset['features_combined'])

# %% Splitting
if not do_test:
    train_features, test_features, train_labels, test_labels = train_test_split(
        dataset['features_combined'],
        dataset['labels'],
        test_size=0.2,
        random_state=42)
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
    if not do_test or not do_ensemble:
        test_predictions = svm_classifier.predict(test_features)
        accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
        print(f'--- SVM --- \nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
        [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]

# %% Random Forest
if do_RF:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=300, criterion='gini', n_jobs=-1)

    rf_classifier.fit(train_features, train_labels)
    if not do_test or do_ensemble:
        test_predictions = rf_classifier.predict(test_features)
        accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
        print(f'--- RF --- \nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
        [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]

# %% Extreme Gradient Boosting
if do_XGB:
    import xgboost as xgb
    xgb_classifier = xgb.XGBClassifier(booster='gbtree', max_depth=6)

    xgb_classifier.fit(train_features, train_labels)
    if not do_test or do_ensemble:
        test_predictions = xgb_classifier.predict(test_features)
        accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
        print(f'--- XGBosst --- \nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
        [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]

# %% Linear Discriminant Analysis
if do_LDA:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_classifier = LinearDiscriminantAnalysis(solver='svd')

    lda_classifier.fit(train_features, train_labels)
    if not do_test or do_ensemble:
        test_predictions = lda_classifier.predict(test_features)
        accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
        print(f'--- LDA --- \nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
        [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]

# %% Ensemble
if do_ensemble:
    estimators = []
    if do_SVM: estimators.append(('svm', svm_classifier))
    if do_RF: estimators.append(('rf', rf_classifier))
    if do_XGB: estimators.append(('xgb', xgb_classifier))
    if do_LDA: estimators.append(('lda', lda_classifier))

    voting_classifier = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    voting_classifier.fit(train_features, train_labels)
    test_predictions = voting_classifier.predict(test_features)

    accuracy, f1, recall, precision, splitted_metrics = get_metrics(test_labels, test_predictions)
    print(f'--- Ensemble --- \nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
    [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]

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
'''data = np.array(test_predictions)
integer_counts = Counter(data)
for number, count in integer_counts.items():
    print(f"{number}: {count} times")'''

# %%