import scipy.io as io
import numpy as np
import os
import copy

from model.detection_model.TextSnake_pytorch.dataset.data_util import pil_load_img
from model.detection_model.TextSnake_pytorch.dataset.dataload import TextDataset, TextInstance
from model.detection_model.TextSnake_pytorch.dataset.read_json import read_json, read_dict

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

        # self.image_root = os.path.join(data_root, 'train_images')
        self.image_root = os.path.join(data_root, 'train_images' if is_training else 'test_images')
        # self.image_root = os.path.join(data_root, 'crop_images_new')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_path = os.path.join(data_root, 'train_labels.json')
        # self.annotation_path = os.path.join(data_root, 'crop_result_js.json')
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

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path,
                                      image_shape=image_shape)

    def __len__(self):
        return len(self.image_list)