import numpy as np
import json

from model.detection_model.TextSnake_pytorch.dataset.dataload import TextInstance


# def read_dict(data, img_name):
#     polygons = []
#     for poly in data[img_name.replace('.jpg', '')]:
#         pts = np.array(poly['points']).astype(np.int32)
#         ori = 'c'
#         text = poly['transcription']
#         illegibility = poly['illegibility']
#
#         # for ArT dataset, key 'language' exists
#         # for LSVT dataset, key 'language' do not exist
#         language = None
#         if 'language' in poly:
#             language = poly['language']
#         polygons.append(TextInstance(pts, ori, text, illegibility, language))
#     return polygons


def read_dict(data, img_name):
    # poly = data[int(img_name.lstrip('gt_').rstrip('.jpg').split('_')[1])]
    poly = data[img_name.replace('.jpg', '')]

    result = []
    for instance in poly:
        pts = np.array(instance['points']).astype(np.int32)
        ori = 'c'
        text = instance['transcription']
        illegibility = instance['illegibility']

        # for ArT dataset, key 'language' exists
        # for LSVT dataset, key 'language' do not exist
        language = None
        if 'language' in poly:
            language = instance['language']

        result.append(TextInstance(pts, ori, text, illegibility, language))

    return result


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


