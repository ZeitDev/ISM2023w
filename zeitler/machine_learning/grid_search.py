'''Desc: 1:1 replication of the paper: https://univ-angers.hal.science/hal-03846675/document'''
# %%
import os
import csv
import cv2
import copy
import numpy as np
import pandas as pd
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

import xgboost as xgb

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

def feature_extraction(image):
    image = skimage.filters.unsharp_mask(image, radius=0.5, amount=2)

    image_rgb = (image*255).astype('uint8')
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
        #if len(dataset['images']) == 200: break

    return dataset

def z_score_normalization(values):
    mean_value = np.mean(values)
    std_deviation = np.std(values)
    normalized_values = (values - mean_value) / std_deviation

    return normalized_values

def get_normalization_values(values):
    mean_value = np.mean(values)
    std_deviation = np.std(values)

    return [mean_value, std_deviation]

def normalize_dataset(dataset):
    for feature in dataset['features']:
        dataset['features'][feature] = z_score_normalization(dataset['features'][feature])

    dataset['features_combined'] = []
    for i in range(len(dataset['images'])):
        features_combined = []
        for feature in dataset['features']:
            features_combined.append(dataset['features'][feature][i])
        dataset['features_combined'].append(features_combined)

    return dataset

def get_vector_features(dataset):
    temp = {'hist': [], 'edges': [], 'dft_red': [], 'dft_green': [], 'cH_red': [], 'cH_green': []}
    for i in tqdm(range(len(dataset['images']))):
        calc_features = calc_vector_features(dataset['images'][i])
        for feature in calc_features:
            temp[feature].extend(calc_features[feature])

    normalization_values = {'hist': [], 'edges': [], 'dft_red': [], 'dft_green': [], 'cH_red': [], 'cH_green': []}
    for feature in temp:
        normalization_values[feature].extend(get_normalization_values(temp[feature]))

    for i in tqdm(range(len(dataset['images']))):
        calc_features = calc_vector_features(dataset['images'][i])
        for feature in calc_features:
            calc_features[feature] = (calc_features[feature] - normalization_values[feature][0]) / normalization_values[feature][1]
            dataset['features_combined'][i].extend(list(calc_features[feature].flatten()))

    return dataset

def calc_vector_features(image):
    image_rgb = (skimage.filters.unsharp_mask(image, radius=0.5, amount=2) * 255).astype('uint8')
    image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')
    
    # Color Histogram
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])

    # Edge Detection
    edges = cv2.Canny(image_gray, 50, 150)

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

    return {'hist': hist, 'edges': edges, 'dft_red': dft_red, 'dft_green': dft_green, 'cH_red': cH_red, 'cH_green': cH_green}

    #features = list(hist.flatten()) + list(normalized_edges.flatten())
    #dataset['features_combined'][i].extend(list(features))

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

def modelfit(alg, dtrain_features, dtrain_labels, dtest_features, dtest_labels, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain_features, label=dtrain_labels)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    alg.fit(dtrain_features, dtrain_labels, eval_metric='auc')
    dtest_predictions = alg.predict(dtest_features)
        
    accuracy, f1, recall, precision, splitted_metrics = get_metrics(dtest_labels, dtest_predictions)
    print(f'\nAccuracy: {accuracy} \nF1: {f1} \nRecall: {recall} \nPrecision: {precision}')
    [print(f'{label_metrics}') for label_metrics in splitted_metrics.items()]


# %% SETTINGS
do_test = False
sampling = False

# %% Datatset
if 'dataset' not in locals():
    dataset = create_dataset(os.path.join('ISM', 'train'))
    dataset = normalize_dataset(dataset)
    dataset = get_vector_features(dataset)

    dataset['labels'] = np.array(dataset['labels'])
    dataset['features_combined'] = np.array(dataset['features_combined'])
    del dataset['images']
    del dataset['features']

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

# %% Extreme Gradient Boosting
xgb1 = xgb.XGBClassifier(
    booster='gbtree',
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    objective='multi:softmax',
    num_class=4)

modelfit(xgb1, train_features, train_labels, test_features, test_labels)


# %%
