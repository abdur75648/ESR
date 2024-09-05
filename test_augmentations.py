import os
import random
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

def random_flip(image, mode='horizontal'):
    if mode == 'horizontal':
        return np.fliplr(image)
    elif mode == 'vertical':
        return np.flipud(image)
    else:
        raise ValueError("Mode should be 'horizontal' or 'vertical'.")

def random_rotation(image, angle_range=(-15, 15)):
    angle = random.uniform(angle_range[0], angle_range[1])
    return rotate_image(image, angle)

def rotate_image(image, angle):
    h, w, _ = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def color_jitter(image, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25):
    # Convert float32 [0,1] to uint8 [0,255]
    image_uint8 = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)

    # Define ColorJitter transform
    cj = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                saturation=saturation, hue=hue)
    jittered = cj(pil_image)

    # Convert back to float32 [0,1]
    jittered_np = np.array(jittered).astype(np.float32) / 255.0
    return jittered_np

def gaussian_blur(image, kernel_size=random.choice([(3, 3), (5, 5), (7, 7)])):
    return cv2.GaussianBlur(image, kernel_size, 0)

def add_gaussian_noise(image, mean=0, sigma=0.05):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

# Function to save images
def save_image(image, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert float32 [0,1] to uint8 [0,255]
    image_uint8 = (image * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV if needed
    cv2.imwrite(save_path, cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))

# Main testing function
def test_augmentations(sample_image_path, output_dir, crop_size=256):
    # Load image using OpenCV and convert to RGB
    img = cv2.imread(sample_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {sample_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Visualize original image
    save_image(img, os.path.join(output_dir, "original.png"))

    # 1. Random Horizontal Flip
    flipped_h = random_flip(img, mode='horizontal')
    save_image(flipped_h, os.path.join(output_dir, "flip_horizontal.png"))

    # 2. Random Vertical Flip
    flipped_v = random_flip(img, mode='vertical')
    save_image(flipped_v, os.path.join(output_dir, "flip_vertical.png"))

    # 3. Random Rotation
    rotated = random_rotation(img)
    save_image(rotated, os.path.join(output_dir, "rotation.png"))

    # 4. Color Jittering
    jittered = color_jitter(img)
    save_image(jittered, os.path.join(output_dir, "color_jitter.png"))

    # 5. Gaussian Blur
    blurred = gaussian_blur(img)
    save_image(blurred, os.path.join(output_dir, "gaussian_blur.png"))

    # 6. Add Gaussian Noise
    noisy = add_gaussian_noise(img)
    save_image(noisy, os.path.join(output_dir, "gaussian_noise.png"))

    # 7. Add Poisson Noise
    noisy_poisson = add_poisson_noise(img)
    save_image(noisy_poisson, os.path.join(output_dir, "poisson_noise.png"))

# Example usage
if __name__ == "__main__":
    # Path to the sample image to test
    sample_image_path = 'test_img/1.png'

    # Directory where augmented images will be saved
    output_dir = 'augmented_images'

    # Size for random cropping
    img = cv2.imread(sample_image_path)
    crop_size = int(min(img.shape[0], img.shape[1]) * 0.9)

    # Run the augmentation tests
    test_augmentations(sample_image_path, output_dir, crop_size)
