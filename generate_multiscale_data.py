import os
import glob
import random
import argparse
from tqdm import tqdm
from PIL import Image, ImageFilter
import numpy as np

def apply_random_resizing(img, target_size=256):
    # Resize to a random size between 128 and 256 and then back to 256x256
    random_size = random.randint(128, 256)
    img = img.resize((random_size, random_size), resample=Image.LANCZOS)
    img = img.resize((target_size, target_size), resample=Image.LANCZOS)
    return img

def apply_gaussian_blur(img):
    # Apply a Gaussian blur with a random radius
    radius = random.uniform(0.5, 2.0)  # Adjust range as needed
    return img.filter(ImageFilter.GaussianBlur(radius))

def apply_additive_noise(img):
    # Add random Gaussian noise
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def main(args):
    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in tqdm(path_list):
        basename = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path)
        width, height = img.size
        assert width == height == 1024, 'Image must be 1024x1024 - Current size: {}x{}'.format(width, height)

        # Apply random resizing, Gaussian blur, and additive noise
        img = apply_random_resizing(img)
        if random.random() < 0.5:
            img = apply_gaussian_blur(img)
        if random.random() < 0.5:
            img = apply_additive_noise(img)

        img.save(os.path.join(args.output, f'{basename}T0.png'))

if __name__ == '__main__':
    """
    Generate degraded images for super-resolution training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
