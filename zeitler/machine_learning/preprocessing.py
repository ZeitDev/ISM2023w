# %%
import os
import csv
import time
import numpy as np
import histomicstk as htk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io, exposure, filters, transform

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration

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
        image = io.imread(os.path.join(self.images_location, self.image_filenames[index]))
        label = self.labels[os.path.splitext(self.image_filenames[index])[0]]

        if self.transform: image = self.transform(image)

        return image, label

def show(image, mask_out):
    vals = np.random.rand(256, 3)
    vals[0, ...] = [0.9, 0.9, 0.9]
    cMap = ListedColormap(1 - vals)

    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image)
    ax[1].imshow(mask_out, cmap=cMap)
    plt.show()

# %%
train_dataset = Dataset('train', 'train.csv', transform=None)

# %%
image = train_dataset[0][0]

# %%
image = exposure.equalize_hist(image)
#show(image, image_normalizied)

# %%
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}
W_target = np.array([
    [0.5807549,  0.08314027,  0.08213795],
    [0.71681094,  0.90081588,  0.41999816],
    [0.38588316,  0.42616716, -0.90380025]
])

mask_out, _ = get_tissue_mask(
    image, deconvolve_first=True,
    n_thresholding_steps=1, sigma=1.5, min_size=30)
mask_out = transform.resize(
    mask_out == 0, output_shape=image.shape[:2],
    order=0, preserve_range=True) == 1

show(image, mask_out)

# %%
image_normalizied = reinhard(image, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'], mask_out=mask_out)
show(image, image_normalizied)

# %%
stain_unmixing_routine_params = {
    'stains': ['hematoxylin', 'eosin'],
    'stain_unmixing_method': 'macenko_pca',
}
image_normalizied = deconvolution_based_normalization(image, W_target=W_target, stain_unmixing_routine_params=stain_unmixing_routine_params, mask_out=mask_out)
show(image, image_normalizied)

# %%
