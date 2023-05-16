import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt


image_directory = "path/afhq_cat"
image_paths = glob.glob(image_directory + "/*.jpg")  

resized_images = []

for path in image_paths:
    img = Image.open(path)
    img = img.resize((64, 64), Image.BILINEAR)  # Resize the image to 64x64 pixels using bilinear interpolation
    resized_images.append(img)
    
# flatten the resized images.
flattened_images = []

for img in resized_images:
    flattened_img = np.array(img).reshape(-1, 3)  # Flatten the image to a 4096x3 matrix
    flattened_images.append(flattened_img)

# Stack the flattened images to create the 3-D array.
X = np.stack(flattened_images)  # Stack the flattened images to create a 3-D array of size 5653x4096x3

X_0 = X[:, :, 0]  # Extract the red channel matrix X_0 (5653x4096)
X_1 = X[:, :, 1]  # Extract the green channel matrix (5653x4096)
X_2 = X[:, :, 2]  # Extract the blue channel matrix (5653x4096)

def apply_pca(X_i):
    
    n = X_i.shape[0]
    # As we stated one of the assumptions of PCA is centered data. So, we are going to center our data. 
    X_i_centered = X_i - np.mean(X_i, axis=0)
    
    # Now we are writing the covariance matrix
    cov_matrix = (1 / n) * X_i_centered.T @ X_i_centered

    # As stated now the eigenvectors and eigenvalues of the covariance matrix are going to be calculated.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the first 10 principal components
    principal_components = sorted_eigenvectors[:, :10]

    # Calculate PVE
    pve = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    # Calculate the cumulative PVE
    cumulative_pve = np.cumsum(pve)

    return principal_components, pve, cumulative_pve, np.array(eigenvectors)
  
red_principals, red_pve, red_cumulative_pve, eigen_red = apply_pca(X_0)
green_principals, green_pve, green_cumulative_pve, eigen_green = apply_pca(X_1)
blue_principals, blue_pve, blue_cumulative_pve, eigen_blue = apply_pca(X_2)

for i in range(10):
    print(f"Red Channel, PC{i+1}: PVE = {red_pve[i]}, Cumulative PVE = {red_cumulative_pve[i]}")
    print(f"Green Channel, PC{i+1}: PVE = {green_pve[i]}, Cumulative PVE = {green_cumulative_pve[i]}")
    print(f"Blue Channel, PC{i+1}: PVE = {blue_pve[i]}, Cumulative PVE = {blue_cumulative_pve[i]}")
    
def find_min_components(cumulative_pve):
    min_components = np.argmax(cumulative_pve >= 0.7) + 1
    return min_components

min_red_components = find_min_components(red_cumulative_pve)
min_green_components = find_min_components(green_cumulative_pve)
min_blue_components = find_min_components(blue_cumulative_pve)

print(f"Minimum number of Red Channel components for 70% PVE: {min_red_components}")
print(f"Minimum number of Green Channel components for 70% PVE: {min_green_components}")
print(f"Minimum number of Blue Channel components for 70% PVE: {min_blue_components}")

def draw_scree_plot(pve, channel_name, num_components):
    x = np.arange(1, num_components + 1)
    y = pve[:num_components]

    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel('Principal Components')
    plt.ylabel('Proportion of Variance Explained (PVE)')
    plt.title(f'Scree Plot - {channel_name} Channel')
    plt.xticks(np.arange(1, num_components + 1))
    plt.grid(True)
    plt.show()

# Draw scree plots for each channel
draw_scree_plot(red_pve, 'Red', 20)

plt.clf()
draw_scree_plot(green_pve, 'Green', 20)

plt.clf()
draw_scree_plot(blue_pve, 'Blue', 20)

red_images = red_principals[:, :10].reshape(-1, 64, 64)
green_images = green_principals[:, :10].reshape(-1, 64, 64)
blue_images = blue_principals[:, :10].reshape(-1, 64, 64)

red_images = (red_images - np.min(red_images)) / (np.max(red_images) - np.min(red_images))
green_images = (green_images - np.min(green_images)) / (np.max(green_images) - np.min(green_images))
blue_images = (blue_images - np.min(blue_images)) / (np.max(blue_images) - np.min(blue_images))

visuals = np.stack([red_images, green_images, blue_images], axis=3)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(visuals[i])
    ax.axis('off')
    ax.set_title(f'PC{i+1}')
plt.tight_layout()
plt.show()

image_path = "path/afhq_cat/flickr_cat_000003.jpg"
img = Image.open(image_path)
img = img.resize((64, 64), Image.BILINEAR)

flattened_img = np.array(img).reshape(-1, 3)

flattened_img_red = flattened_img[:, 0].astype(np.float32)
flattened_img_green = flattened_img[:, 1].astype(np.float32)
flattened_img_blue = flattened_img[:, 2].astype(np.float32)

flattened_img_red_centered = flattened_img_red - flattened_img_red.mean()
flattened_img_green_centered = flattened_img_green - flattened_img_green.mean()
flattened_img_blue_centered = flattened_img_blue - flattened_img_blue.mean()

def reconstruct_image(flattened_img, flattened_img_centered, eigen_vectors, k):
    reconstructed = np.dot(eigen_vectors[:k], flattened_img_centered)
    reconstructed_image = np.dot(eigen_vectors[:k].T, reconstructed) + flattened_img.mean()
    reconstructed_image = np.reshape(reconstructed_image, (64, 64))
    add_image = Image.fromarray(np.uint8(reconstructed_image))
    return add_image
  
k_values = [1, 50, 250, 500, 1000, 4096]
num_plots = len(k_values)
fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

for i, k in enumerate(k_values):
    part_red = reconstruct_image(flattened_img_red, flattened_img_red_centered, eigen_red, k)
    part_green = reconstruct_image(flattened_img_green, flattened_img_green_centered, eigen_green, k)
    part_blue = reconstruct_image(flattened_img_blue, flattened_img_blue_centered, eigen_blue, k)

    reconstructed_image = np.stack((np.array(part_red), np.array(part_green), np.array(part_blue)), axis=-1)
    reconstructed_image = Image.fromarray(np.uint8(reconstructed_image))

    axes[i].imshow(reconstructed_image)
    axes[i].set_title(f"k = {k}")
    axes[i].axis('off')

plt.show()
