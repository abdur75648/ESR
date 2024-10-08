import os
import cv2, random
import numpy as np
import torchvision.transforms as transforms
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize

def rotate_image(image, angle):
    h, w, _ = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def random_blur(image, kernel_size=random.choice([(3, 3), (3, 3), (5, 5), (7,7)])):
    return cv2.GaussianBlur(image, kernel_size, 0)

def random_resize(image):
    scale_factor = random.uniform(0.5, 1.0)
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT])
    resized_image = cv2.resize(resized_image, (image.shape[1], image.shape[0]), interpolation=interpolation)
    return resized_image

def add_gaussian_noise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 1).astype(np.float32)
    return noisy_image

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def augment_image(img_gt, img_lq):
    # print("Augmenting image")
    # Random Horizontal Flip
    if random.random() < 0.5:
        img_gt = np.fliplr(img_gt)
        img_lq = np.fliplr(img_lq)

    # # Random Rotation
    # if random.random() < 0.1:
    #     angle = random.uniform(-5, 5)
    #     img_gt = rotate_image(img_gt, angle)
    #     img_lq = rotate_image(img_lq, angle)

    # Color Jittering
    if random.random() < 0.25:
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        img_lq_uint8 = (img_lq * 255).astype(np.uint8)
        img_lq_uint8 = np.array(color_jitter(transforms.ToPILImage()(img_lq_uint8)))
        img_lq = img_lq_uint8.astype(np.float32) / 255.0

    # Random Blur
    if random.random() < 0.25:
        img_lq = random_blur(img_lq)

    # Random Resize
    if random.random() < 0.25:
        img_lq = random_resize(img_lq)

    # Add Gaussian Noise
    if random.random() < 0.25:
        img_lq = add_gaussian_noise(img_lq)

    # # Add Poisson Noise
    # if random.random() < 0.5:
    #     img_lq = add_poisson_noise(img_lq)

    return img_gt, img_lq

@DATASET_REGISTRY.register()
class RealESRGANPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealESRGANPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image

            # ## Old Version ## File Format 1 ## gt_path, lq_path
            # # gt/00000001.png, img_512/00000001.png
            # # gt/00000002.png, img_512/00000002.png
            # # gt/00000003.png, img_512/00000003.png
            # with open(self.opt['meta_info']) as fin:
            #     paths = [line.strip() for line in fin]
            # self.paths = []
            # self.coords = None # No coordinates in previous version
            # for path in paths:
            #     gt_path, lq_path = path.split(', ')
            #     gt_path = os.path.join(self.gt_folder, gt_path)
            #     lq_path = os.path.join(self.lq_folder, lq_path)
            #     self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))

            ## New Version ## File Format 2 ## gt_path, lq_path, x, y, w, h
            # gt/00000001.png, img_512/00000001.png, 268, 180, 463, 463
            # gt/00000002.png, img_512/00000002.png, 272, 179, 455, 455
            # gt/00000003.png, img_512/00000003.png, 274, 179, 451, 451
            with open(self.opt['meta_info']) as fin:
                lines = [line.strip() for line in fin]
            self.paths = []
            self.coords = []
            for line in lines:
                gt_path, lq_path, x, y, w, h = line.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
                self.coords.append([int(x), int(y), int(w), int(h)])

        else:
            raise NotImplementedError('Only support meta_info')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        gt_size = self.opt['gt_size']
        assert img_gt.shape[:2] == (gt_size, gt_size), f"Error: GT image size {img_gt.shape[:2]} is not the same as gt_size {gt_size}"

        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq = augment_image(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'coords': self.coords[index]}

    def __len__(self):
        return len(self.paths)
