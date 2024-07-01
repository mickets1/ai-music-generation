import cv2
import numpy as np

# Function to blend two images using a cosine alpha for smoother transition
def blend_images_cosine(image1, image2, size):
    alpha = np.linspace(0, 1, size)
    alpha = (1 - np.cos(alpha * np.pi)) / 2  # Cosine alpha for smoother transition
    blended = np.zeros_like(image1)
    for i in range(size):
        blended[:, i, :] = cv2.addWeighted(image1[:, i, :], alpha[i], image2[:, i, :], 1 - alpha[i], 0)
    return blended

# Function to apply blending region to the image
def apply_blended_region(image1, image2, blended_region, blend_region_size):
    image1[:, -blend_region_size:, :] = blended_region
    image2[:, :blend_region_size, :] = blended_region
    return image1, image2

# Function to split image into frequency bands
def split_into_bands(image, num_bands=5):
    height, width, _ = image.shape
    band_height = height // num_bands
    bands = [image[i*band_height:(i+1)*band_height, :, :] for i in range(num_bands)]
    return bands

# Function to recombine frequency bands into a single image
def recombine_bands(bands):
    return np.vstack(bands)

# Load the images (ensure they are 512x512 RGB images)
image_paths = [
    'EXAMPLES/RGB.png', 
    'EXAMPLES/RGB.png'
]  # Add all image paths here
images = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_paths]

# Define the blending region size
blend_region_size = 10  # Adjust the size of the blending region as needed

# Number of transition frames
num_frames = 5  # Adjust the number of frames as needed for smooth transition

# Number of frequency bands
num_bands = 10  # Adjust the number of frequency bands as needed

# Process each pair of images in the sequence
for idx in range(len(images) - 1):
    image1 = images[idx]
    image2 = images[idx + 1]
    
    # Split images into frequency bands
    bands_image1 = split_into_bands(image1, num_bands)
    bands_image2 = split_into_bands(image2, num_bands)
    
    blended_bands1 = []
    blended_bands2 = []
    
    for band1, band2 in zip(bands_image1, bands_image2):
        # Extract the regions to blend
        end_region_band1 = band1[:, -blend_region_size:, :]
        begin_region_band2 = band2[:, :blend_region_size, :]
        
        # Generate the blended region using cosine transition
        blended_region = blend_images_cosine(end_region_band1, begin_region_band2, blend_region_size)
        
        # Replace the original regions with blended regions
        blended_band1, blended_band2 = apply_blended_region(band1.copy(), band2.copy(), blended_region, blend_region_size)
        
        blended_bands1.append(blended_band1)
        blended_bands2.append(blended_band2)
    
    # Recombine frequency bands into a single image
    blended_image1 = recombine_bands(blended_bands1)
    blended_image2 = recombine_bands(blended_bands2)
    
    # Save the blended images
    for i in range(num_frames + 1):
        cv2.imwrite(f'INPUT_FOLDER/blended_image_{idx:02d}_{i:02d}.png', cv2.cvtColor(blended_image1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'INPUT_FOLDER/blended_image_{idx+1:02d}_{i:02d}.png', cv2.cvtColor(blended_image2, cv2.COLOR_RGB2BGR))