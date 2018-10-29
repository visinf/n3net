'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import os
import PIL.Image as Image

import torch.utils.data as data
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

import utils

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ToGrayscale(object):
    def __call__(self, img):
        return img.convert('L', (0.2989, 0.5870, 0.1140, 0))

class MaybeFlip(object):
    def __call__(self, img):
        if img.size[1] > img.size[0]:
            img.transpose(Image.TRANSPOSE)


def make_dataset(dir, filter=None, depth=-1):
    images = []
    if filter is None:
        filter = lambda x: True
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(utils.walklevel(dir, level=depth)):
        for fname in sorted(fnames):
            if is_image_file(fname) and filter(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    return images

class PlainImageFolder(data.Dataset):
    r"""
    Adapted from torchvision.datasets.folder.ImageFolder
    """

    def __init__(self, root, transform=None,
                 loader=default_loader, cache=False, filter=None, depth=-1):
        self.cache = cache
        self.img_cache = {}
        if isinstance(root, list):
            imgs = []
            for r in root:
                imgs.extend(make_dataset(r, filter=filter, depth=depth))
        else:
            imgs = make_dataset(root, filter=filter, depth=depth)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target)
        """
        path = self.imgs[index]
        if not index in self.img_cache:
            img = self.loader(path)
            if self.cache:
                self.img_cache[index] = img
        else:
            img = self.img_cache[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)