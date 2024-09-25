import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import multiprocessing
from functools import partial

def apply_target_resizing(img, target_size):
    random_scale = 0.75 + 0.5 * np.random.rand()
    img = img.resize((int(img.width * random_scale), int(img.height * random_scale)), resample=Image.Resampling.LANCZOS)
    img = img.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
    return img

def process_image(path, output_dir, target_size, input_size=1024):
    basename = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path)
    width, height = img.size
    assert width == height == input_size, f'Image size should be {input_size}x{input_size}, but got {width}x{height}'

    # Apply resizing
    img = apply_target_resizing(img, target_size=target_size)

    # Save the image
    output_path = os.path.join(output_dir, f'{basename}.png')
    img.save(output_path)

def main(args):
    # Get list of all images in the input directory
    path_list = sorted(glob.glob(os.path.join(args.input, '*')))

    # Prepare the processing function with required arguments
    process_func = partial(process_image, output_dir=args.output, target_size=args.target_size, input_size=args.input_size)

    # Use multiprocessing Pool to parallelize the work
    cpu_count = min(32,os.cpu_count())
    with multiprocessing.Pool(cpu_count) as pool:
        list(tqdm(pool.imap(process_func, path_list), total=len(path_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--input_size', type=int, default=1024, help='Input size of the image')
    parser.add_argument('--target_size', type=int, default=512, help='Target size of the image')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
