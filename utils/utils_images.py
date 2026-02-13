import os
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None 
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def _imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ext = ext if ext else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def save_single_image(sample, save_path, scale=4):
    img = sample[0].permute(1, 2, 0).squeeze().clamp(0, 1).cpu().numpy()

    if img.ndim == 2:
        ndarr = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        if scale != 1:
            new_width = int(ndarr.shape[1] * scale)
            new_height = int(ndarr.shape[0] * scale)
            ndarr = cv2.resize(ndarr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        _imwrite_unicode(save_path, ndarr)
        return

    ndarr = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    if scale != 1:
        new_width = int(ndarr.shape[1] * scale)
        new_height = int(ndarr.shape[0] * scale)
        ndarr = cv2.resize(ndarr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    bgr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    _imwrite_unicode(save_path, bgr)


def save_grid_image(x_t, save_path, scale=4, normalize=True):
    if normalize:
        x_t = (x_t + 1) / 2
    x_t = x_t.clamp(0, 1)
    nrow = int((x_t.size(0)) ** 0.5)
    grid = make_grid(x_t, nrow=nrow, padding=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    if scale != 1:
        new_width = int(ndarr.shape[1] * scale)
        new_height = int(ndarr.shape[0] * scale)
        ndarr = cv2.resize(ndarr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    bgr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    _imwrite_unicode(save_path, bgr)