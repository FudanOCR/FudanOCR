import torch
import numpy as np
import cv2
import os
from skimage import measure, color
from util.config import config as cfg
from util.misc import fill_hole

def visualize_network_output(output, tr_mask, tcl_mask, prefix):

    tr_pred = output[:, :2]
    tr_score, tr_predict = tr_pred.max(dim=1)

    tcl_pred = output[:, 2:4]
    tcl_score, tcl_predict = tcl_pred.max(dim=1)

    tr_predict = tr_predict.cpu().numpy()
    tcl_predict = tcl_predict.cpu().numpy()

    tr_target = tr_mask.cpu().numpy()
    tcl_target = tcl_mask.cpu().numpy()

    for i in range(len(tr_pred)):
        tr_pred = (tr_predict[i] * 255).astype(np.uint8)
        tr_targ = (tr_target[i] * 255).astype(np.uint8)

        tcl_pred = (tcl_predict[i] * 255).astype(np.uint8)
        tcl_targ = (tcl_target[i] * 255).astype(np.uint8)

        tr_show = np.concatenate([tr_pred, tr_targ], axis=1)
        tcl_show = np.concatenate([tcl_pred, tcl_targ], axis=1)
        show = np.concatenate([tr_show, tcl_show], axis=0)
        show = cv2.resize(show, (512, 512))
        path = os.path.join(cfg.vis_dir, '{}_{}.png'.format(prefix, i))
        cv2.imwrite(path, show)


def visualize_detection(image, tr, tcl, contours, illegal_contours=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)
    if illegal_contours is not None:
        image_show = cv2.polylines(image_show, illegal_contours, True, (0, 255, 0), 3)

    conts, _ = cv2.findContours(tcl.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in conts:
        # remove small regions
        if cv2.contourArea(cont) < 50:
            tcl = cv2.fillPoly(tcl, [cont], 0)

    tr = cv2.cvtColor((tr * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tcl = cv2.cvtColor((tcl * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # labels = measure.label(tcl, connectivity=2)
    # tcl_color = color.label2rgb(labels) * 255

    # # slightly enlarge for easier to get tcl
    # kernel = np.ones((5, 5), np.uint8)
    # tcl_color = cv2.dilate(tcl_color, kernel, iterations=2)

    image_show = np.concatenate([image_show, tr, tcl], axis=1)
    return image_show
    # path = os.path.join(cfg.vis_dir, image_id)
    # cv2.imwrite(path, image_show)


if __name__ == '__main__':
    import json
    import os
    json_path = '.../result.json'
    img_path = '.../test_images'
    files = os.listdir(img_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    for img_name in files:
        image = cv2.imread(os.path.join(img_path, img_name))
        poly = data[img_name.replace('.jpg', '').replace('gt', 'res')]
        pts = np.array(poly['points']).astype(np.int32)
        image_show = cv2.polylines(image, [pts], True, (0, 0, 255), 3)
        cv2.imwrite(img_name, image_show)
