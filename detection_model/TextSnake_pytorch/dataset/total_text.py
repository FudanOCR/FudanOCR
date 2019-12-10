import scipy.io as io
import numpy as np
import os
import copy

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset
from dataset.read_json import read_json, read_dict

class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'train_images' if is_training else 'demo_images')
        #self.image_root = os.path.join(data_root, 'crop_images_new')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_path = os.path.join(data_root, 'train_labels.json')
        #self.annotation_path = os.path.join(data_root, 'crop_result_js.json')
        self.data_dict = read_json(self.annotation_path)

    def __getitem__(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        image_shape = image.shape

        # Read annotation
        polygons = read_dict(self.data_dict, image_id)

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        # Todo: may be bug here
        for i, polygon in enumerate(polygons):
            if not polygon['illegibility']:
                polygon.find_bottom_and_sideline(polygon.points)

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path, image_shape=image_shape)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='/home/shf/fudan_ocr_system/datasets/ICDAR19/',
        ignore_list=None,#'/workspace/mnt/group/ocr/qiutairu/dataset/ArT_train/ignore_list.txt',
        is_training=True,
        transform=transform
    )

    for idx in range(len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[idx]
        if img.shape[0] != 3:
            print(idx, img.shape)

    testset = TotalText(
        data_root='/home/shf/fudan_ocr_system/datasets/ICDAR19/',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=512, mean=means, std=stds)
    )

    for idx in range(len(testset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = testset[idx]
        if img.shape[0] != 3:
            print(idx, img.shape)

    # path = '/workspace/mnt/group/ocr/qiutairu/dataset/ArT_train/train_images'
    # files = os.listdir(path)
    #
    # for file in files:
    #     image = pil_load_img(os.path.join(path, file))
    #     if image.shape[2] != 3:
    #         print(file, image.shape)

    # path = '/workspace/mnt/group/ocr/qiutairu/dataset/ArT_train/test_images'
    # files = os.listdir(path)
    #
    # for file in files:
    #     image = pil_load_img(os.path.join(path, file))
    #     if image.shape[2] != 3:
    #         print(file, image.shape)
