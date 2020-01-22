import json
import numpy as np
from evaluation.polygon_wrapper import iou
from evaluation.polygon_wrapper import iod

input_json_path = '/home/shf/fudan_ocr_system/TextSnake_pytorch/output/result.json'
gt_json_path = '/home/shf/fudan_ocr_system/datasets/ICDAR19/train_labels.json'
# input_json_path = '/workspace/mnt/group/ocr/qiutairu/code/textsnake_pytorch/output/lsvt_0318_test/result.json'
# gt_json_path = '/workspace/mnt/group/ocr/qiutairu/dataset/LSVT_full_train/train_labels.json'
save_img_path = '/home/shf/fudan_ocr_system/TextSnake_pytorch/vis/{}'.format(input_json_path.split('/')[-2])
iou_threshold = 0.5

def input_reading(polygons):
    det = []
    for polygon in polygons:
        polygon['points'] = np.array(polygon['points'])
        det.append(polygon)
    return det


def gt_reading(gt_dict, img_key):
    polygons = gt_dict[img_key]
    gt = []
    for polygon in polygons:
        polygon['points'] = np.array(polygon['points'])
        gt.append(polygon)
    return gt


def detection_filtering(detections, groundtruths, threshold=0.5):
    """ignore detected illegal text region"""
    before_filter_num = len(detections)
    for gt_id, gt in enumerate(groundtruths):
        if (gt['transcription'] == '###') and (gt['points'].shape[1] > 1):
            gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
            gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
            for det_id, detection in enumerate(detections):
                det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                det_y = list(map(int, np.squeeze(detection['points'][:, 1])))
                det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]

    # if before_filter_num - len(detections) > 0:
    #     print("Ignore {} illegal detections".format(before_filter_num - len(detections)))

    return detections


def gt_filtering(groundtruths):
    before_filter_num = len(groundtruths)
    for gt_id, gt in enumerate(groundtruths):
        if gt['transcription'] == '###' or gt['points'].shape[0] < 3:
            groundtruths[gt_id] = []

    groundtruths[:] = [item for item in groundtruths if item != []]

    # if before_filter_num - len(groundtruths) > 0:
    #     print("Ignore {} illegal groundtruths".format(before_filter_num - len(groundtruths)))

    return groundtruths


if __name__ == '__main__':
    # Initial config
    global_tp = 0
    global_fp = 0
    global_fn = 0

    # load json file as dict
    with open(input_json_path, 'r') as f:
        input_dict = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_dict = json.load(f)

    false_case = []
    for idx, (input_img_key, input_cnts) in enumerate(input_dict.items()):
        detections = input_reading(input_cnts)
        groundtruths = gt_reading(gt_dict, input_img_key.replace('res', 'gt'))
        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
        groundtruths = gt_filtering(groundtruths)

        iou_table = np.zeros((len(groundtruths), len(detections)))
        det_flag = np.zeros((len(detections), 1))
        gt_flag = np.zeros((len(groundtruths), 1))
        tp = 0
        fp = 0
        fn = 0
        for gt_id, gt in enumerate(groundtruths):
            gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
            gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                    det_y = list(map(int, np.squeeze(detection['points'][:, 1])))

                    iou_table[gt_id, det_id] = iou(det_x, det_y, gt_x, gt_y)

                best_matched_det_id = np.argmax(
                    iou_table[gt_id, :])  # identified the best matched detection candidates with current groundtruth

                matched_id = np.where(iou_table[gt_id, :] >= iou_threshold)
                if iou_table[gt_id, best_matched_det_id] >= iou_threshold:
                    if matched_id[0].shape[0] < 2:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                    else:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                        # if there are more than 1 matched detection, only 1 is contributed to tp, the rest are fp
                        fp = fp + (matched_id[0].shape[0] - 1.0)

        # Update local and global tp, fp, and fn
        inv_gt_flag = 1 - gt_flag
        fn = np.sum(inv_gt_flag)
        inv_det_flag = 1 - det_flag
        fp = fp + np.sum(inv_det_flag)

        global_fp = global_fp + fp
        global_fn = global_fn + fn
        if tp + fp == 0:
            local_precision = 0
        else:
            local_precision = tp / (tp + fp)

        if tp + fn == 0:
            local_recall = 0
        else:
            local_recall = tp / (tp + fn)

        global_precision_now = global_tp / (global_tp + global_fp)
        global_recall_now = global_tp / (global_tp + global_fn)
        f_score_now = 2 * global_precision_now * global_recall_now / (global_precision_now + global_recall_now)

        print('[ {0:4} ] / [ {1:4} ] {2:12} Precision: {3:.4f}, Recall: {4:.4f}'.format(idx+1, len(input_dict),
                                                                                        input_img_key.replace('res', 'gt') + '.jpg',
                                                                                        local_precision, local_recall) +
              ', GlobalP now: {:.4f}, GlobalR now: {:.4f}, GlobalF now: {:.4f}'.format(global_precision_now,
                                                                                       global_recall_now, f_score_now))
        if local_precision == 0 or local_recall == 0 or 2 * local_precision * local_recall / (local_precision + local_recall) < 0.3:
            false_case.append((input_img_key.replace('res', 'gt') + '.jpg', local_precision, local_recall))

    global_precision = global_tp / (global_tp + global_fp)
    global_recall = global_tp / (global_tp + global_fn)
    f_score = 2 * global_precision * global_recall / (global_precision + global_recall)

    print('Global Precision: {:.4f}, Recall: {:.4f}, F_score: {:.4f}\n'.format(global_precision, global_recall, f_score))
    print('False case: {}'.format(len(false_case)))
    import shutil
    import os
    if not os.path.exists(os.path.join(save_img_path, 'false_case')):
        os.mkdir(os.path.join(save_img_path, 'false_case'))
    for item in false_case:
        print('{}, {:.4f}, {:.4f}'.format(item[0], item[1], item[2]))
        shutil.copyfile(os.path.join(save_img_path, item[0]),
                        os.path.join(save_img_path, 'false_case', '{}_{:.2f}_{:.2f}.jpg'.format(item[0].replace('.jpg', ''), item[1], item[2])))
    print('over')























