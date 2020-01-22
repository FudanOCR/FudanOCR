import os
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from imgaug import augmenters as iaa

import config as cfg
from utils.preprocess import Anno


class custom_dset(data.Dataset):
    '''
    Preprocess images, load ground-truth, and build dataset.
    '''

    def __init__(self, split):

        if split == 'train':
            self.img_list = sorted([img_name for img_name in os.listdir(cfg.train_img)], key=by_id)
            self.gt_list = sorted([gt_name for gt_name in os.listdir(cfg.train_gt)], key=by_id1)
            # self.gt_list = sorted([gt_name for gt_name in os.listdir(cfg.train_gt)], key=by_id)
            print('Found %d images in train set.' % len(self.img_list))
            self.img_dir = cfg.train_img

        if split == 'val':
            self.img_list = sorted([img_name for img_name in os.listdir(cfg.val_img)], key=by_id)
            self.gt_list = sorted([gt_name for gt_name in os.listdir(cfg.val_gt)], key=by_id1)
            # self.gt_list = sorted([gt_name for gt_name in os.listdir(cfg.val_gt)], key=by_id)
            print('Found %d images in val set.' % len(self.img_list))
            self.img_dir = cfg.val_img

        print('Loading %s set annotation' % split)
        anno = Anno(self.img_list, self.img_dir, split)
        anno.check_anno()
        self.anno_dir, self.nimg_dir = anno.get_dir()
        self.split = split

    def __getitem__(self, index):
        img_name = self.img_list[index]

        with Image.open(os.path.join(self.nimg_dir, img_name[:-4] + '.jpg')).convert('RGB') as img:
            img = np.asarray(img)
        gt = np.load(os.path.join(self.anno_dir, img_name[:-4] + '.npy'))

        if False:  # if self.split is 'train':
            # Augmentation
            flip_ = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2)
            ], random_order=True)
            flip = flip_.to_deterministic()

            img = flip.augment_image(img)
            gt = flip.augment_image(gt)

        img = transform(img.copy())
        gt = torch.from_numpy(gt.copy())

        return img, gt

    def __len__(self):
        return len(self.img_list)

def by_id1(name):
    '''Sort list by image_id.'''
    return int(name[7:-4])

def by_id(name):
    '''Sort list by image_id.'''
    return int(name[4:-4])
    # return int(name[3:-4])


def collate_fn(batch):
    '''Merge samples to form a mini-batch.'''
    img, gt = zip(*batch)
    bs = len(gt)
    images, gts = [], []
    for i in range(bs):
        images.append(img[i])
        gts.append(gt[i])
    images = torch.stack(images, 0)
    gts = torch.stack(gts, 0)
    return images, gts


def transform(img):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = Compose([ToTensor(), normalize])
    return trans(img)
