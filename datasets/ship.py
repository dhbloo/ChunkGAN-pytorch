import torch
import numpy as np
import PIL
import os
import cv2
from torchvision import transforms
from utils.labelmap import convert_labelmap, extract_bounding_boxes
from . import BaseDataset

try:
    import pyspng
except ImportError:
    pyspng = None


class Ship(BaseDataset):
    def __init__(self,
                 data_root,
                 resolution=None,
                 min_object_area=0.01,
                 max_object_area=0.2,
                 gamma=1.0,
                 blur_radius=0.1,
                 thresh_type='global',
                 thresh_min=30,
                 thresh_C=30,
                 block_size=0.1,
                 morph_iteration=None,
                 morph_kernel_size=0.005,
                 min_contrast=100):
        super(BaseDataset, self).__init__()
        assert os.path.isdir(data_root)
        self.path = data_root

        all_fnames = [
            os.path.relpath(os.path.join(root, fname), start=self.path)
            for root, _dirs, files in os.walk(self.path) for fname in files
        ]
        file_ext = lambda fname: os.path.splitext(fname)[1].lower()

        PIL.Image.init()
        self.image_fnames = sorted(fname for fname in all_fnames
                                   if file_ext(fname) in PIL.Image.EXTENSION)
        if len(self.image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.raw_shape = [len(self.image_fnames)] + list(self._load_raw_image(0).shape)

        if resolution is not None and (self.raw_shape[2] != resolution
                                       or self.raw_shape[3] != resolution):
            self.shape = self.raw_shape[0::3] + [resolution, resolution]
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.shape[2], self.shape[3]))])
        else:
            self.shape = self.raw_shape[0::3] + self.raw_shape[1:3]
            self.transform = transforms.ToTensor()

        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
        self.gamma = gamma
        self.blur_radius = blur_radius
        self.thresh_type = thresh_type
        self.thresh_min = thresh_min
        self.thresh_C = thresh_C
        self.block_size = block_size
        self.morph_iteration = morph_iteration
        self.morph_kernel_size = morph_kernel_size
        self.min_contrast = min_contrast

    def _load_raw_image(self, idx):
        fname = self.image_fnames[idx]
        with open(os.path.join(self.path, fname), 'rb') as f:
            if pyspng is not None and os.path.splitext(fname)[1].lower() == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        return image

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        image = self._load_raw_image(idx)
        numpy_image = image.squeeze()
        labelmap = convert_labelmap(numpy_image,
                                    labelmap_size=(self.shape[2], self.shape[3]),
                                    gamma=self.gamma,
                                    blur_radius=self.blur_radius,
                                    thresh_type=self.thresh_type,
                                    thresh_min=self.thresh_min,
                                    thresh_C=self.thresh_C,
                                    block_size=self.block_size,
                                    morph_iteration=self.morph_iteration,
                                    morph_kernel_size=self.morph_kernel_size)
        bboxes = extract_bounding_boxes(labelmap, self.min_object_area, self.max_object_area)

        def bboxes_filter(rect):
            return rect.max() - rect.min() >= self.min_contrast

        if numpy_image.shape != labelmap.shape:
            numpy_image = cv2.resize(numpy_image, labelmap.shape, interpolation=cv2.INTER_LANCZOS4)
        bboxes = [(x, y, w, h) for x, y, w, h in bboxes
                  if bboxes_filter(numpy_image[y:y + h, x:x + w])]
        image = self.transform(image)

        return image, bboxes

    def collect_fn(self, batch):
        """
        Since each image may have a different number of objects, 
        we need a collate function (to be passed to the DataLoader).

        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes
        """
        images = list()
        chunks = list()
        for b in batch:
            images.append(b[0])
            chunks.append(b[1])
        images = torch.stack(images, dim=0)
        return images, chunks

    @property
    def image_shape(self):
        return self.shape[1:]

    @property
    def description(self):
        return f'Ship Dataset (data_root={self.path})'
