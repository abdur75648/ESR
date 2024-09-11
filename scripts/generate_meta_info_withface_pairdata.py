import argparse
import glob
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Load the face cascade model once globally
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(args):
    img_path_gt, img_path_lq, root_gt, root_lq = args
    img_name_gt = os.path.relpath(img_path_gt, root_gt)
    img_name_lq = os.path.relpath(img_path_lq, root_lq)

    # Read the image once, and convert to grayscale for face detection
    img = cv2.imread(img_path_gt)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        return f'{img_name_gt}, {img_name_lq}, {x}, {y}, {w}, {h}'
    else:
        return f'{img_name_gt}, {img_name_lq}, No face detected'

def main(args):
    # Prepare the file paths
    img_paths_gt = sorted(glob.glob(os.path.join(args.input[0], '*')))
    img_paths_lq = sorted(glob.glob(os.path.join(args.input[1], '*')))

    # Check that the GT and LQ folders have the same number of images
    assert len(img_paths_gt) == len(img_paths_lq), ('GT folder and LQ folder should have the same length, but got '
                                                    f'{len(img_paths_gt)} and {len(img_paths_lq)}.')

    # Pack arguments in a tuple for multiprocessing
    tasks = [(img_path_gt, img_path_lq, args.root[0], args.root[1]) for img_path_gt, img_path_lq in zip(img_paths_gt, img_paths_lq)]

    # Use all available CPU cores for multiprocessing
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_image, tasks), total=len(tasks)))

    # Write the results to a text file
    with open(args.meta_info_file, 'w') as txt_file:
        for result in results:
            txt_file.write(result + '\n')


if __name__ == '__main__':
    """This script is used to generate meta info (txt file) for paired images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['datasets/Prasanna/gt', 'datasets/Prasanna/img_512'],
        help='Input folder, should be [gt_folder, lq_folder]')
    parser.add_argument('--root', nargs='+', default=[None, None], help='Folder root, will use the ')
    parser.add_argument(
        '--meta_info_file',
        type=str,
        required=True,
        help='txt path for meta info')
    args = parser.parse_args()

    assert len(args.input) == 2, 'Input folder should have two elements: gt folder and lq folder'
    assert len(args.root) == 2, 'Root path should have two elements: root for gt folder and lq folder'
    os.makedirs(os.path.dirname(args.meta_info_file), exist_ok=True)
    #  if meta_info_file exists and is not empty, clean it
    if os.path.exists(args.meta_info_file):
        with open(args.meta_info_file, 'r') as f:
            if len(f.readlines()) > 0:
                with open(args.meta_info_file, 'w') as f:
                    f.write('')
    for i in range(2):
        if args.input[i].endswith('/'):
            args.input[i] = args.input[i][:-1]
        if args.root[i] is None:
            args.root[i] = os.path.dirname(args.input[i])

    main(args)
