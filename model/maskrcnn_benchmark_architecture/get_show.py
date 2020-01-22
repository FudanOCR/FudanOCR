import cv2
import os
import json
import numpy as np

def draw_lines(img, points, is_gt=False):
    points = np.array(points, np.int)
    num_of_pts = points.shape[0]

    color = (0, 255, 0) if is_gt else (0, 0, 255)

    for i in range(num_of_pts):
        cv2.line(img, (points[i%num_of_pts, 0], points[i%num_of_pts, 1]), (points[(i+1)%num_of_pts, 0], points[(i+1)%num_of_pts, 1]), color, 2)

    return img


gt_file = ''
res_file = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0180000/res.json' # Fscore: 71.83
gt_file = '../datasets/LSVT/train_full_labels.json'
image_dir = '../datasets/LSVT/train_full_images_0/train_full_images_0/'

gt_annos = json.load(open(gt_file, 'r'))
res_annos = json.load(open(res_file, 'r'))

res_keys = list(res_annos.keys())

for k in res_keys:
    im_path = os.path.join(image_dir, k + '.jpg')
    img = cv2.imread(im_path)

    gt_list = gt_annos[k]
    res_list = res_annos[k]

    for gt_anno in gt_list:
        gt_pts = gt_anno['points']
        if not gt_anno['illegibility']:
            draw_lines(img, gt_pts, True)

    for res_anno in res_list:
        res_pts = res_anno['points']
        draw_lines(img, res_pts, False)

    print('impath:', im_path)
    print('img:', img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(0)