import os
import glob
import argparse
from PIL import Image
import multiprocessing
from functools import partial
from tqdm import tqdm

# Function to crop the image as per the requirement
def crop_image(image_path, output_dir):
    basename = os.path.basename(image_path)
    img = Image.open(image_path)
    width, height = img.size

    # Ensure the image is 2160x2160 as expected
    assert width == 2160 and height == 2160, f"Image size should be 2160x2160, but got {width}x{height}"

    # Vertical crop: Start from the top and keep the first 1600 pixels
    top_crop = 1600

    # Horizontal crop: Ignore 280 pixels from both left and right, keeping the middle part of 1600 pixels
    left_crop = 280
    right_crop = 1600+280

    # Perform the cropping
    cropped_img = img.crop((left_crop, 0, right_crop, top_crop))

    # Resize the cropped image to 1024x1024
    cropped_img = cropped_img.resize((1024, 1024), Image.LANCZOS)

    # Save the cropped image to the output directory
    cropped_img.save(os.path.join(output_dir, basename))

# Main function to handle multiprocessing and the cropping process
def main(args):
    # Get a list of all PNG images in the input directory
    image_paths = sorted(glob.glob(os.path.join(args.input, '*.png')))

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Prepare the partial function to pass additional arguments to the processing function
    process_func = partial(crop_image, output_dir=args.output)

    # Use multiprocessing Pool to crop images in parallel
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap(process_func, image_paths), total=len(image_paths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop 2160x2160 images to 1024x1024 and save them.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input directory with PNG images.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for cropped images.')

    args = parser.parse_args()

    # Run the main function
    main(args)
