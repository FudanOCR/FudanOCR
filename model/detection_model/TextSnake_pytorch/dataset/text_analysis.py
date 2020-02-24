import cv2
import os
import numpy as np
from model.detection_model.TextSnake_pytorch.dataset.read_json import read_json, read_dict


def recorder(record, ratio):
    ranges = [key.split('~') for key in record.keys()]
    for range in ranges:
        if int(range[0]) <= ratio < int(range[1]):
            record['{}~{}'.format(range[0], range[1])] += 1
            break
    return record


if __name__ == '__main__':
    path = '/home/shf/fudan_ocr_system/datasets/ICDAR19/'
    json_name = 'train_labels.json'
    maxlen = 1280

    train_files = os.listdir(os.path.join(path, 'train_images'))
    test_files = os.listdir(os.path.join(path, 'test_images'))
    data_dict = read_json(os.path.join(path, json_name))

    for idx, file in enumerate(train_files):
        polygons = read_dict(data_dict, file)

        for polygon in polygons:
            if polygon.language == 'Latin' and ' ' in polygon.text:
                print(file)
                break

    print("\nTest Images\n")

    for idx, file in enumerate(test_files):
        polygons = read_dict(data_dict, file)

        for polygon in polygons:
            if polygon.language == 'Latin' and ' ' in polygon.text:
                print(file)
                break
