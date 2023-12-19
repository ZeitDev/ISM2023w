# %%
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#import histomicstk as htk

import torch
import skimage

# %% 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_location, labels_location, transform=None):
        self.images_location = os.path.join('data', images_location)
        self.labels_location = os.path.join('data', labels_location)
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
        label = self.labels[os.path.splitext(self.image_filenames[index])[0]]

        if self.transform: image = self.transform(image)

        return image, label

def show(image, image2):
    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image)
    ax[1].imshow(image2)
    plt.show()

# %% Image Preprocessing

train_dataset = Dataset('train', 'train.csv', transform=None)
image = train_dataset[0][0]
image_resized = skimage.transform.resize(image, (200, 200))
image_sharpened = skimage.filters.unsharp_mask(image_resized, radius=2, amount=5)

show(image, image_sharpened)

# %% First Order Statistics
image = image_sharpened

mean = np.mean(image)
standard_deviation = np.std(image)
median = np.median(image)
percentile_25 = np.percentile(image, 25)
percentile_50 = np.percentile(image, 50)
percentile_75 = np.percentile(image, 75)

print('Mean: ', mean, '\nStandard Deviation: ', standard_deviation, '\nMedian: ', median, '\nPercentile 25: ', percentile_25, '\nPercentile 50: ', percentile_50, '\nPercentile 75: ', percentile_75)

# %% GLCM - Gray Level Co-occurrence Matrix
image_gray = skimage.color.rgb2gray(image)
image_uint = image_gray.astype('uint8')

glcm = skimage.feature.graycomatrix(image_uint, [1], [0, np.pi/4, 2*np.pi/4, 3*np.pi/4])

contrast = skimage.feature.graycoprops(glcm, 'contrast')
dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')
homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
energy = skimage.feature.graycoprops(glcm, 'energy')
correlation = skimage.feature.graycoprops(glcm, 'correlation')
asm = skimage.feature.graycoprops(glcm, 'ASM')

print('Contrast: ', contrast, '\nDissimilarity: ', dissimilarity, '\nHomogeneity: ', homogeneity, '\nEnergy: ', energy, '\nCorrelation: ', correlation, '\nASM: ', asm)


# %% Hu Invariant Moments

_, image_binary = cv2.threshold(image_gray*255, 128, 255, cv2.THRESH_BINARY)
moments = cv2.moments(image_binary)
hu_moments = cv2.HuMoments(moments)
log_hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))

for i in range(7): print(f"Hu Moment {i+1}: {log_hu_moments[i][0]}")

# %%
