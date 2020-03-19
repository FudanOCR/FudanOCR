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
        # segm
        res_name = result['image_id'].replace('gt_', 'res_').replace('.jpg', '')
        mask_conf = result['score']
        res_points, res_area = to_poly(result['segmentation'])
        res_size = result['segmentation']['size']

        if not (len(res_points) > 0 and len(res_points[0]) > 0):
            continue

        # bbox
        bbox = coco_result['bbox'][idx]['bbox']
        bbox_conf = coco_result['bbox'][idx]['score']
        res_bbox = xywha_to_xyxy(bbox).astype(np.int32).tolist()

        # init
        if res_name not in output_result:
            output_result[res_name] = []
            bbox_result[res_name] = []

        output_result[res_name].append({
            "points": res_points,
            "confidence": mask_conf,
            'area': res_area,
            'size': res_size
        })

        bbox_result[res_name].append({
            "points": res_bbox,
            "confidence": bbox_conf
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
        for j in range(i, nums):
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
    dets = []
    for kk in keep:
        dets.append({
            'points': bbox_infos[kk],
            'confidence': scores[kk]
        })
    return dets


def xywha_to_xyxy(rect):
    cx, cy, w, h, angle = rect
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]

    pts = [lt, rt, rb, lb]
    angle = -angle

    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)[:, :2]

    return rotated_pts.astype(np.int32)


def rotate_pts(pts, rotate_ct, angle):
    pts = [pt+[1] for pt in pts]
    cx, cy = rotate_ct
    angle = -angle

    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)[:, :2]

    return rotated_pts.astype(np.int32)
