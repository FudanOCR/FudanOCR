def detval():
    import numpy as np
    import json
    from Evaluation.polygon_wrapper import iod
    from Evaluation.polygon_wrapper import area_of_intersection
    from Evaluation.polygon_wrapper import area
    from util.config import config
    import os

    input_json_path = os.path.join(config.output_dir, 'result.json')#'/home/shf/fudan_ocr_system/TextSnake_pytorch/output/result.json'
    gt_json_path = os.path.join(config.data_root, config.dataset, 'train_labels.json')#'/home/shf/fudan_ocr_system/datasets/ICDAR19/train_labels.json'


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
        """
        ignore detected illegal text region
        """
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

        if before_filter_num - len(detections) > 0:
            print("Ignore {} illegal detections".format(before_filter_num - len(detections)))

        return detections


    def gt_filtering(groundtruths):
        before_filter_num = len(groundtruths)
        for gt_id, gt in enumerate(groundtruths):
            if gt['transcription'] == '###' or gt['points'].shape[0] < 3:
                groundtruths[gt_id] = []

        groundtruths[:] = [item for item in groundtruths if item != []]

        if before_filter_num - len(groundtruths) > 0:
            print("Ignore {} illegal groundtruths".format(before_filter_num - len(groundtruths)))

        return groundtruths


    def sigma_calculation(det_x, det_y, gt_x, gt_y):
        """
        sigma = inter_area / gt_area
        """
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)


    def tau_calculation(det_x, det_y, gt_x, gt_y):
        """
        tau = inter_area / det_area
        """
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(det_x, det_y)), 2)


    def one_to_one(local_sigma_table, local_tau_table, local_accumulative_recall,
                   local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                   gt_flag, det_flag):
        """

        Args:
            local_sigma_table:
            local_tau_table:
            local_accumulative_recall:
            local_accumulative_precision:
            global_accumulative_recall:
            global_accumulative_precision:
            gt_flag:
            det_flag:

        Returns:

        """
        for gt_id in range(num_gt):
            qualified_sigma_candidates = np.where(local_sigma_table[gt_id, :] > tr)
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
            qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > tp)
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]


            if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
                global_accumulative_recall = global_accumulative_recall + 1.0
                global_accumulative_precision = global_accumulative_precision + 1.0
                local_accumulative_recall = local_accumulative_recall + 1.0
                local_accumulative_precision = local_accumulative_precision + 1.0

                gt_flag[0, gt_id] = 1
                matched_det_id = np.where(local_sigma_table[gt_id, :] > tr)
                det_flag[0, matched_det_id] = 1
        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


    def one_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                   local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                   gt_flag, det_flag):
        for gt_id in range(num_gt):
            # skip the following if the groundtruth was matched
            if gt_flag[0, gt_id] > 0:
                continue

            non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)
            num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]

            if num_non_zero_in_sigma >= k:
                # search for all detections that overlaps with this groundtruth
                qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= tp) & (det_flag[0, :] == 0))
                num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

                if num_qualified_tau_candidates == 1:
                    if local_tau_table[gt_id, qualified_tau_candidates] >= tp and local_sigma_table[gt_id, qualified_tau_candidates] >= tr:
                        # became an one-to-one case
                        global_accumulative_recall = global_accumulative_recall + 1.0
                        global_accumulative_precision = global_accumulative_precision + 1.0
                        local_accumulative_recall = local_accumulative_recall + 1.0
                        local_accumulative_precision = local_accumulative_precision + 1.0

                        gt_flag[0, gt_id] = 1
                        det_flag[0, qualified_tau_candidates] = 1
                elif np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= tr:
                    gt_flag[0, gt_id] = 1
                    det_flag[0, qualified_tau_candidates] = 1

                    global_accumulative_recall = global_accumulative_recall + fsc_k
                    global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * fsc_k

                    local_accumulative_recall = local_accumulative_recall + fsc_k
                    local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * fsc_k

        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


    def many_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                   local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                   gt_flag, det_flag):
        for det_id in range(num_det):
            # skip the following if the detection was matched
            if det_flag[0, det_id] > 0:
                continue

            non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
            num_non_zero_in_tau = non_zero_in_tau[0].shape[0]

            if num_non_zero_in_tau >= k:
                # search for all detections that overlaps with this groundtruth
                qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
                num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

                if num_qualified_sigma_candidates == 1:
                    if local_tau_table[qualified_sigma_candidates, det_id] >= tp and local_sigma_table[qualified_sigma_candidates, det_id] >= tr:
                        # became an one-to-one case
                        global_accumulative_recall = global_accumulative_recall + 1.0
                        global_accumulative_precision = global_accumulative_precision + 1.0
                        local_accumulative_recall = local_accumulative_recall + 1.0
                        local_accumulative_precision = local_accumulative_precision + 1.0

                        gt_flag[0, qualified_sigma_candidates] = 1
                        det_flag[0, det_id] = 1
                elif np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp:
                    det_flag[0, det_id] = 1
                    gt_flag[0, qualified_sigma_candidates] = 1

                    global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                    global_accumulative_precision = global_accumulative_precision + fsc_k

                    local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                    local_accumulative_precision = local_accumulative_precision + fsc_k
        return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag



    # Initial config
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_sigma = []
    global_tau = []
    tr = 0.7
    tp = 0.6
    fsc_k = 0.8
    k = 2

    # load json file as dict
    with open(input_json_path, 'r') as f:
        input_dict = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_dict = json.load(f)

    for input_img_key, input_cnts in input_dict.items():
        print(input_img_key)
        detections = input_reading(input_cnts)
        groundtruths = gt_reading(gt_dict, input_img_key.replace('res', 'gt'))
        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
        groundtruths = gt_filtering(groundtruths)

        local_sigma_table = np.zeros((len(groundtruths), len(detections)))
        local_tau_table = np.zeros((len(groundtruths), len(detections)))

        for gt_id, gt in enumerate(groundtruths):
            if len(detections) > 0:
                gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
                gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
                for det_id, detection in enumerate(detections):
                    det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                    det_y = list(map(int, np.squeeze(detection['points'][:, 1])))

                    local_sigma_table[gt_id, det_id] = sigma_calculation(det_x, det_y, gt_x, gt_y)
                    local_tau_table[gt_id, det_id] = tau_calculation(det_x, det_y, gt_x, gt_y)

        global_sigma.append(local_sigma_table)
        global_tau.append(local_tau_table)

    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_det = 0

    print('############## Evaluate Result ###############')
    input_list = list(input_dict.keys())
    for idx in range(len(global_sigma)):
        local_sigma_table = global_sigma[idx]
        local_tau_table = global_tau[idx]

        num_gt = local_sigma_table.shape[0]
        num_det = local_sigma_table.shape[1]

        total_num_gt = total_num_gt + num_gt
        total_num_det = total_num_det + num_det

        local_accumulative_recall = 0
        local_accumulative_precision = 0
        gt_flag = np.zeros((1, num_gt))
        det_flag = np.zeros((1, num_det))

        #######first check for one-to-one case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = one_to_one(local_sigma_table, local_tau_table,
                                      local_accumulative_recall, local_accumulative_precision,
                                      global_accumulative_recall, global_accumulative_precision,
                                      gt_flag, det_flag)

        #######then check for one-to-many case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = one_to_many(local_sigma_table, local_tau_table,
                                       local_accumulative_recall, local_accumulative_precision,
                                       global_accumulative_recall, global_accumulative_precision,
                                       gt_flag, det_flag)

        #######then check for many-to-many case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = many_to_many(local_sigma_table, local_tau_table,
                                        local_accumulative_recall, local_accumulative_precision,
                                        global_accumulative_recall, global_accumulative_precision,
                                        gt_flag, det_flag)

        # print each image evaluate result
        try:
            local_precision = local_accumulative_precision / num_det
        except ZeroDivisionError:
            local_precision = 0

        try:
            local_recall = local_accumulative_recall / num_gt
        except ZeroDivisionError:
            local_recall = 0

        print('{0:12} Precision: {1:.4f}, Recall: {2:.4f}'.format(input_list[idx].replace('res', 'gt') + '.jpg',
                                                                  local_precision, local_recall))

    # print global evaluate result
    try:
        recall = global_accumulative_recall / total_num_gt
    except ZeroDivisionError:
        recall = 0

    try:
        precision = global_accumulative_precision / total_num_det
    except ZeroDivisionError:
        precision = 0

    try:
        f_score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f_score = 0

    print('Global Precision: {:.4f}, Recall: {:.4f}'.format(precision, recall))

    print('over')
