# %%
import os
import cv2
import numpy as np

def calculate_mean_std(images_path, batch_size):
    # Initialize variables to store cumulative sum
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    total_images = 0

    # Iterate through images in batches
    for i in range(0, len(images_path), batch_size):
        batch_images = []
        # Read and preprocess images in the batch
        for j in range(i, min(i + batch_size, len(images_path))):
            image = cv2.imread(images_path[j])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = image / 255.0  # Normalize to [0, 1]
            batch_images.append(image)

        # Calculate mean and std for the batch
        batch_images = np.array(batch_images)
        batch_mean = np.mean(batch_images, axis=(0, 1, 2))
        batch_std = np.std(batch_images, axis=(0, 1, 2))

        # Update cumulative sum
        mean_sum += batch_mean * len(batch_images)
        std_sum += batch_std * len(batch_images)
        total_images += len(batch_images)

    # Calculate final mean and std
    final_mean = mean_sum / total_images
    final_std = std_sum / total_images

    return final_mean, final_std

# Example usage
batch_size = 50  # Adjust the batch size based on your available memory


# Example usage:
path = os.path.join('..', 'data', 'ISM', 'train')
image_paths = os.listdir(path)
image_paths = [os.path.join(path, image_path) for image_path in image_paths]
mean, std = calculate_mean_std(image_paths, batch_size)

print("Mean:", mean)
print("Standard Deviation:", std)
# %%
