import numpy as np
import cv2
from skimage.measure import find_contours
import pycocotools.mask as maskUtils


def to_poly(rle):
    mask = maskUtils.decode(rle)
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    area = float(maskUtils.area(rle))
    if len(contours) == 0:
        return [[]], area
    poly = np.fliplr(contours[0]).astype(np.int32).tolist()

    return poly, area


def coco_results_to_contest(coco_result):
    output_result = {}
    bbox_result = {}
    for idx, result in enumerate(coco_result['segm']):
        print("[ {} ] / [ {} ]".format(idx+1, len(coco_result['segm'])))
        # segm
        res_name = result['image_id'].replace('gt_', 'res_').replace('.jpg', '')
        res_conf = result['score']
        res_points, res_area = to_poly(result['segmentation'])
        res_size = result['segmentation']['size']

        if not (len(res_points) > 0 and len(res_points[0]) > 0):
            continue

        # bbox
        x, y, w, h = cv2.boundingRect(np.array(res_points).astype(np.int32))
        res_bbox = [[int(x), int(y)], [int(x)+int(w), int(y)],
                    [int(x)+int(w), int(y)+int(h)], [int(x), int(y)+int(h)]]

        # init
        if res_name not in output_result:
            output_result[res_name] = []
            bbox_result[res_name] = []

        output_result[res_name].append({
            "points": res_points,
            "confidence": res_conf,
            'area': res_area,
            'size': res_size
        })

        bbox_result[res_name].append({
            "points": res_bbox,
            "confidence": res_conf
        })

    return output_result, bbox_result


def get_mask(box, shape):
    """根据box获取对应的掩膜"""
    tmp_mask = np.zeros(shape, dtype="uint8")
    tmp = np.array(box, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(tmp_mask, [tmp], 255)
#     tmp_mask=cv2.bitwise_and(tmp_mask,mask)
    return tmp_mask, cv2.countNonZero(tmp_mask)


def comput_mmi(area_a, area_b, intersect):
    """
    计算MMI,2018.11.23 add
    :param area_a: 实例文本a的mask的面积
    :param area_b: 实例文本b的mask的面积
    :param intersect: 实例文本a和实例文本b的相交面积
    :return:
    """
    eps = 1e-5
    if area_a == 0 or area_b == 0:
        area_a += eps
        area_b += eps
        print("the area of text is 0")
    return max(float(intersect)/area_a, float(intersect)/area_b)


def mask_nms(polygons, shape, mmi_thres=0.5, conf_thres=0.4):
    """
    mask nms 实现函数
    :param polygons: 检测结果，[{'points':[[],[],[]],'confidence':int},{},{}]
    :param shape: 当前检测的图片原大小
    :param mmi_thres: 检测的阈值
    :param conf_thres: 检测的阈值
    """
    # 获取bbox及对应的score
    bbox_infos = []
    areas = []
    scores = []
    for poly in polygons:
        if poly['confidence'] > conf_thres:
            bbox_infos.append(poly['points'])
            areas.append(poly['area'])
            scores.append(poly['confidence'])
    # print('before ',len(bbox_infos))
    keep = []
    # order = np.array(scores).argsort()[::-1]
    order = np.array(areas).argsort()[::-1]
    # print("order:{}".format(order))
    nums = len(bbox_infos)
    suppressed = np.zeros(nums, dtype=np.int)
    # print("lens:{}".format(nums))

    # 循环遍历
    for i in range(nums):
        idx = order[i]
        if suppressed[idx] == 1:
            continue
        keep.append(idx)
        mask_a, area_a = get_mask(bbox_infos[idx], shape)

        # child_masks = []
        for j in range(i+1, nums):
            idx_j = order[j]
            if suppressed[idx_j] == 1:
                continue
            mask_b, area_b = get_mask(bbox_infos[idx_j], shape)

            # 获取两个文本的相交面积
            merge_mask = cv2.bitwise_and(mask_a, mask_b)
            area_intersect = cv2.countNonZero(merge_mask)

            # 计算MMI
            mmi = comput_mmi(area_a, area_b, area_intersect)
            # print("area_a:{},area_b:{},inte:{},mmi:{}".format(area_a,area_b,area_intersect,mmi))

            # if mmi >= 0.95:
            #     child_masks.append(idx_j)
            # elif mmi >= mmi_thres:
            if mmi >= mmi_thres:
                suppressed[idx_j] = 1
                or_mask = cv2.bitwise_or(mask_a, mask_b)
                sum_area = cv2.countNonZero(or_mask)
                padded_mask = np.zeros((or_mask.shape[0] + 2, or_mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = or_mask
                contours = find_contours(padded_mask, 0.5)
                poly = np.fliplr(contours[0]).astype(np.int32).tolist()
                bbox_infos[idx] = poly
                areas[idx] = sum_area

        # # split parent to multiple child
        # child_scores = [scores[child_idx] for child_idx in child_masks]
        # # if len(child_masks) == 1 and max(child_scores) > scores[idx]:
        # #     # add child to keep
        # #     child_idx = child_masks[0]
        # #     suppressed[child_idx] = 1
        # #     keep.append(child_idx)
        # #     # # get sub mask (parent - child)
        # #     # child_mask, child_area = get_mask(bbox_infos[child_idx], shape)
        # #     # sub_mask = cv2.subtract(mask_a, child_mask)
        # #     # sub_area = cv2.countNonZero(sub_mask)
        # #     # padded_sub_mask = np.zeros((sub_mask.shape[0] + 2, sub_mask.shape[1] + 2), dtype=np.uint8)
        # #     # padded_sub_mask[1:-1, 1:-1] = sub_mask
        # #     # # update parent
        # #     # contours = find_contours(padded_sub_mask, 0.5)
        # #     # poly = np.fliplr(contours[0]).astype(np.int32).tolist()
        # #     # bbox_infos[idx] = poly
        # #     # areas[idx] = sub_area
        #
        # if len(child_masks) > 1 and max(child_scores) > scores[idx]:
        #     # suppress parent
        #     suppressed[idx] = 1
        #     keep.pop()
        #     # add child to keep
        #     for child_idx in child_masks:
        #         suppressed[child_idx] = 1
        #         keep.append(child_idx)

    dets = []
    for kk in keep:
        dets.append({
            'points': bbox_infos[kk],
            'confidence': scores[kk]
        })
    return dets
