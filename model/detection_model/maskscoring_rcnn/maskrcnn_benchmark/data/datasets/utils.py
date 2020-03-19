import numpy as np
from PIL import Image
import json
import cv2


def read_info(data, img_name):
    img = data[img_name.replace('.jpg', '')]
    if 'img' in img_name:
        img_id = int(img_name.replace('.jpg', '').replace('gt_img', ''))
    else:
        img_id = int(img_name.replace('.jpg', '').replace('gt_', ''))

    return {
        'license': 1,
        'file_name': img_name,
        'width': int(img['width']),
        'height': int(img['height']),
        'id': img_id
    }


def read_anno(data, img_name):
    polygons = []
    for poly in data[img_name.replace('.jpg', '')]['polygons']:
        pts = np.array(poly['points']).astype(np.int32)
        if 'transcription' in poly:
            text = poly['transcription']
        else:
            text = poly['transcriprions']
        illegibility = poly['illegibility']

        # for ArT dataset, key 'language' exists
        # for LSVT dataset, key 'language' do not exist
        language = None
        if 'language' in poly:
            language = poly['language']

        polygons.append({
            "points": pts,
            "text": text,
            "illegibility": illegibility,
            "language": language
        })

    return polygons


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def pil_load_img(path):
    image = Image.open(path)

    # Expand the dim if gray
    if image.mode is not 'RGB':
        image = image.convert('RGB')

    return image


def generate_bbox(segmentation, shape):
    H, W = shape
    pts = np.array(segmentation).astype(np.int32)
    x_min = min(pts[:, 0]) if min(pts[:, 0]) >= 0 else 0
    x_max = max(pts[:, 0]) if max(pts[:, 0]) < W else W-1
    y_min = min(pts[:, 1]) if min(pts[:, 1]) >= 0 else 0
    y_max = max(pts[:, 1]) if max(pts[:, 1]) < H else H-1

    return [x_min, y_min, x_max-x_min, y_max-y_min]


def generate_rbox(segmentation, shape):
    H, W = shape
    pts = np.array(segmentation).astype(np.int32)
    rect = cv2.minAreaRect(pts)
    gt_ind = np.int0(cv2.boxPoints(rect)).reshape(-1, 8)[0]
    pt1 = (int(gt_ind[0]), int(gt_ind[1]))
    pt2 = (int(gt_ind[2]), int(gt_ind[3]))
    pt3 = (int(gt_ind[4]), int(gt_ind[5]))
    pt4 = (int(gt_ind[6]), int(gt_ind[7]))

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    if edge1 > edge2:
        width = edge1
        height = edge2
        if pt1[0] - pt2[0] != 0:
            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    else:
        width = edge2
        height = edge1
        if pt2[0] - pt3[0] != 0:
            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    if angle < -45.0:
        angle = angle + 180

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2

    return [x_ctr, y_ctr, width, height, angle]


def get_img_id(img_name):
    if 'img' in img_name:
        return int(img_name.replace('gt_img', '').replace('.jpg', ''))
    else:
        return int(img_name.replace('gt_', '').replace('.jpg', ''))

