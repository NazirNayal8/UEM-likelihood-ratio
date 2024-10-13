"""
Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
"""

import os
import numpy as np
import albumentations as A
import random
from typing import Optional, Callable
from torch.utils.data import Dataset
from PIL import Image

class COCO(Dataset):
    # train_id_in = 0
    # train_id_out = 254
    # min_image_size = 480

    def __init__(
            self, 
            root: str, 
            split: str = "train",
            version: str = "ood_seg",
            transform: Optional[Callable] = None, 
            size_constraint=None
        ) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
        self.size_constraint = size_constraint
        
        labels_folder = os.path.join(self.root, "annotations", f"{version}_" + self.split)
        for root, _, filenames in os.walk(labels_folder):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year, "trainval" + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    self.targets.append(os.path.join(root, filename))
                    self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and ground truth in PIL format or as torch tensor"""
        image = np.array(Image.open(self.images[i]).convert('RGB'))
        target = np.array(Image.open(self.targets[i]).convert('L'))
        if self.size_constraint is not None:
            H, W = image.shape[:2]
            if H % self.size_constraint != 0 or W % self.size_constraint != 0:
                H = H - H % self.size_constraint + 14
                W = W - W % self.size_constraint + 14
                aug = A.Resize(H, W)(image=image, mask=target)
                image, target = aug["image"], aug["mask"]
        if self.transform is not None:
            aug = self.transform(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""

        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()
    

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].

    Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def mix_object(current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask, ood_label):
    """
    Adapted from Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
    """
    mask = (cut_object_mask == ood_label)
    
    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    ood_boxes = ood_boxes[0, :]  # (y1, x1, y2, x2)
    y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
    cut_object_mask = cut_object_mask[y1:y2, x1:x2]
    cut_object_image = cut_object_image[y1:y2, x1:x2, :]

    mask = cut_object_mask == ood_label

    idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

    # if current_labeled_mask.shape[0] != 1024 or current_labeled_mask.shape[1] != 2048:
    #     print('wrong size')
    #     print(current_labeled_mask.shape)
    #     return current_labeled_image, current_labeled_mask

    if mask.shape[0] != 0:
        if current_labeled_mask.shape[0] - cut_object_mask.shape[0] < 0 or \
                current_labeled_mask.shape[1] - cut_object_mask.shape[1] < 0:
            # print('wrong size')
            # print(current_labeled_mask.shape)
            return current_labeled_image, current_labeled_mask
        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]
    else:
        # print('no odd pixel to mix')
        h_start_point = 0
        h_end_point = 0
        w_start_point = 0
        w_end_point = 0
    
    result_image = current_labeled_image.copy()
    result_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == ood_label)] = \
        cut_object_image[np.where(idx == ood_label)]
    result_label = current_labeled_mask.copy()
    result_label[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == ood_label)] = \
        cut_object_mask[np.where(cut_object_mask == ood_label)]

    return result_image, result_label

class COCOPastable(Dataset):
    train_id_in = 0
    train_id_out = 254
    min_image_size = 480

    def __init__(
            self, 
            root: str, 
            proxy_size: int, 
            split: str = "train",
            transform: Optional[Callable] = None, 
            shuffle=True
        ) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
        
        for root, _, filenames in os.walk(os.path.join(self.root, "annotations", "ood_seg_" + self.split)):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    self.targets.append(os.path.join(root, filename))
                    self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))

        """
        shuffle data and subsample
        """

        if shuffle:
            zipped = list(zip(self.images, self.targets))
            random.shuffle(zipped)
            self.images, self.targets = zip(*zipped)

        if proxy_size is not None:
            self.images = list(self.images[:int(proxy_size)])
            self.targets = list(self.targets[:int(proxy_size)])
        else:
            self.images = list(self.images[:5000])
            self.targets = list(self.targets[:5000])

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and ground truth in Numpy format or as torch tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return np.array(image), np.array(target)

    def __repr__(self):
        """Return number of images in each dataset."""

        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()