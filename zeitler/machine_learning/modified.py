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
from sklearn.model_selection import GridSearchCV

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
def preprocess_image(image):
    image = skimage.filters.unsharp_mask(image, radius=0.5, amount=2)
    image_rgb = (image*255).astype('uint8')
    image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')

    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the background color
    lower_bound = np.array([0, 0, 0])  # Adjust these values based on your background color
    upper_bound = np.array([10, 10, 10])  # Adjust these values based on your background color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    result = cv2.medianBlur(result, 5)
    lab_image = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel_eq = cv2.equalizeHist(l_channel)
    equalized_image = cv2.merge([l_channel_eq, a_channel, b_channel])
    result = cv2.cvtColor(equalized_image, cv2.COLOR_LAB2RGB)

    image_rgb = result.astype('uint8')
    image_gray = (skimage.color.rgb2gray(image_rgb) * 255).astype('uint8')

    return image_rgb, image_gray

def feature_extraction(image):
    image_rgb, image_gray = preprocess_image(image)

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
# %%
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
        
        #if len(dataset['images']) == 500: break
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
    image_rgb, image_gray = preprocess_image(image)
    
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

# %% SETTINGS
do_z_score_normalization = True
do_unit_normalization = False

do_SVM = False
do_RF = False
do_XGB = True
do_LDA = False

do_pca = False
do_ensemble = False

do_test = True

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
    xgb_classifier = xgb.XGBClassifier(booster='gbtree', max_depth=8, device='gpu', learning_rate=0.3)

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
data = np.array(train_labels)
integer_counts = Counter(data)
for number, count in integer_counts.items():
    print(f"{number}: {count} times")

# %%