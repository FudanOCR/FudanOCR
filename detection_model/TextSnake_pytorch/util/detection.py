import numpy as np
import cv2
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2
import random

class TextDetector(object):

    def __init__(self, tcl_conf_thresh=0.5, tr_conf_thresh=0.3):
        self.tcl_conf_thresh = tcl_conf_thresh
        self.tr_conf_thresh = tr_conf_thresh

    def is_inside(self, img, x, y):
        h, w = img.shape[:2]
        if 0 <= int(x) < w and 0 <= int(y) < h:
            return True
        else:
            return False

    def create_line(self, point1, point2, image):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        h, w = image.shape[:2]

        # y = kx+b
        if not x2 - x1 == 0:
            k = float(y2 - y1) / float(x2 - x1)
            b = y1 - k * x1
        else:
            return np.array([x1, 0]), np.array([x1, h-1])

        def y(_x):
            return int(k*int(_x)+b)

        def x(_y):
            return int((int(_y)-b)/k)

        if y(0) < 0:
            start = [x(0), 0]
        elif y(0) >= h:
            start = [x(h-1), h-1]
        else:
            start = [0, y(0)]

        if y(w-1) < 0 :
            end = [x(0), 0]
        elif y(w-1) >= h:
            end = [x(h-1), h-1]
        else:
            end = [w - 1, y(w - 1)]

        return np.array(start), np.array(end)

    def find_innerpoint(self, cont):
        """
        generate an inner point of input polygon using mean of x coordinate by:
        1. calculate mean of x coordinate(xmean)
        2. calculate maximum and minimum of y coordinate(ymax, ymin)
        2. iterate for each y in range (ymin, ymax), find first segment in the polygon
        3. calculate means of segment
        :param cont: input polygon
        :return:
        """

        xmean = cont[:, 0, 0].mean()
        ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
        found = False
        found_y = []

        for i in np.arange(ymin - 1, ymax + 1, 0.5):
            # if in_poly > 0, (xmean, i) is in `cont`
            in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
            if in_poly > 0:
                found = True
                found_y.append(i)
            # first segment found
            if in_poly < 0 and found:
                break

        if len(found_y) > 0:
            return (xmean, np.array(found_y).mean())

        # if cannot find using above method, try each point's neighbor
        else:
            for p in range(len(cont)):
                point = cont[p, 0]
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        test_pt = point + [i, j]
                        if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
                            return test_pt

    def find_center_innerpoint(self, start, end, cont, contour_img, tcl_pred, center):
        # draw line
        drawing = np.zeros(tcl_pred.shape[:2])
        line = cv2.line(drawing, (start[0], start[1]), (end[0], end[1]), 1, 1)

        cross_points = np.argwhere(line * contour_img)[:, ::-1]

        # if not 2 cross points
        if len(cross_points) < 2:
            return None
        elif len(cross_points) > 2:
            # self.debug_img += line

            dist = [np.linalg.norm(pt - center) for pt in cross_points]
            order = np.argsort(dist)
            # select the nearest one and farthest one
            cross_points = np.array([cross_points[order[0]], cross_points[order[-1]]])

            center_innerpoint = np.mean(cross_points, axis=0)
            return int(center_innerpoint[0]), int(center_innerpoint[1])
        else:
            # self.debug_img += line
            center_innerpoint = np.mean(cross_points, axis=0)
            return int(center_innerpoint[0]), int(center_innerpoint[1])

    def step_forward(self, pt, h=512, w=512, stride=1, direction=1):
        '''
        :param pt: np.array([x, y])
        :param h, w: int
        :param stride: step stride, int
        :param direction: 1 for clockwise, -1 for counter-clockwise
        :return:
        '''
        # avoid corner point
        if direction == 1:
            if pt[0] == 0 and pt[1] == 0:
                pt = np.array([0, 1])
            elif pt[0] == 0 and pt[1] == h - 1:
                pt = np.array([1, h - 1])
            elif pt[0] == w - 1 and pt[1] == h - 1:
                pt = np.array([w - 1, h - 2])
            elif pt[0] == w - 1 and pt[1] == 0:
                pt = np.array([w - 2, 0])
        else:
            if pt[0] == 0 and pt[1] == 0:
                pt = np.array([1, 0])
            elif pt[0] == 0 and pt[1] == h - 1:
                pt = np.array([0, h - 2])
            elif pt[0] == w - 1 and pt[1] == h - 1:
                pt = np.array([w - 2, h - 1])
            elif pt[0] == w - 1 and pt[1] == 0:
                pt = np.array([w - 1, 1])

        new_pt = pt
        if pt[0] == 0:
            new_pt[0] = 0
            new_pt[1] = new_pt[1] + direction * stride
            if new_pt[1] >= h:
                new_pt[1] = h-1
            elif new_pt[1] < 0:
                new_pt[1] = 0
        elif pt[0] == w - 1:
            new_pt[0] = w - 1
            new_pt[1] = new_pt[1] - direction * stride
            if new_pt[1] >= h:
                new_pt[1] = h-1
            elif new_pt[1] < 0:
                new_pt[1] = 0
        elif pt[1] == 0:
            new_pt[0] = new_pt[0] - direction * stride
            new_pt[1] = 0
            if new_pt[0] >= w:
                new_pt[0] = w-1
            elif new_pt[0] < 0:
                new_pt[0] = 0
        elif pt[1] == h - 1:
            new_pt[0] = new_pt[0] + direction * stride
            new_pt[1] = h - 1
            if new_pt[0] >= w:
                new_pt[0] = w-1
            elif new_pt[0] < 0:
                new_pt[0] = 0
        return new_pt

    def find_principal_curve(self, cont, tcl_pred, radii_pred):
        '''
        :param cont: [ [x1, y1], [x2, y2], ...]
        :param tcl_pred: (h, w)
        :param radii_pred: (h, w)
        :return: [ np.array([x1, y1, r1]), np.array([x2, y2, r2]), ... ]
        '''
        h, w = tcl_pred.shape[:2]

        # calc centroid
        M = cv2.moments(cont)
        centroid = np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

        # calc range of x and y
        range_x = np.ptp(cont[:, 0])
        range_y = np.ptp(cont[:, 1])
        if float(range_x) / float(range_y) > 1.1:
            start = np.array([centroid[0], 0])
            end = np.array([centroid[0], h-1])
        elif float(range_y) / float(range_x) > 1.1:
            start = np.array([0, centroid[1]])
            end = np.array([w-1, centroid[1]])
        else:
            def dist_point_to_line(x, y, k, b):
                # line: y=kx+b, x=(y-b)/k
                if y == k*x+b or x == float(y-b)/k:
                    return 0
                dist1 = np.abs(x - float(y-b)/k)
                dist2 = np.abs(y - k*x+b)
                dist3 = np.linalg.norm(np.array([x, k*x+b]) - np.array([float(y-b)/k, y]))
                return float(dist1 * dist2) / dist3
            # y = x + b
            b_45_line = centroid[1] - centroid[0]
            dist_to_45_line = [dist_point_to_line(x, y, 1, b_45_line) for x, y in cont]
            # y = -x + b
            b_135_line = centroid[1] + centroid[0]
            dist_to_135_line = [dist_point_to_line(x, y, -1, b_135_line) for x, y in cont]
            if np.ptp(dist_to_45_line) > np.ptp(dist_to_135_line):
                start, end = self.create_line(centroid, [centroid[0]+1, centroid[1]-1], tcl_pred)
            else:
                start, end = self.create_line(centroid, [centroid[0]+1, centroid[1]+1], tcl_pred)

        # select the point farther from contour as the base point
        base_point = start.copy() if np.linalg.norm(centroid - start) > np.linalg.norm(centroid - end) else end.copy()
        free_point = end.copy() if np.linalg.norm(centroid - start) > np.linalg.norm(centroid - end) else start.copy()

        def is_in_same_axis(pt1, pt2):
            if pt1[0] == 0 and pt2[0] == 0:
                return True
            elif pt1[0] == w-1 and pt2[0] == w-1:
                return True
            elif pt1[1] == 0 and pt2[1] == 0:
                return True
            elif pt1[1] == h-1 and pt2[1] == h-1:
                return True
            else:
                return False

        stride = 16
        principal_curve_cw = []
        principal_curve_ccw = []
        free_point_backup = free_point.copy()

        # draw contour
        drawing = np.zeros(tcl_pred.shape[:2])
        contour_img = cv2.polylines(drawing, [cont], True, 1, 2)    # thickness=2, easier getting cross points
        # self.debug_img += contour_img

        # clockwise forward
        none_count = 0
        while not is_in_same_axis(base_point, free_point):
            ci = self.find_center_innerpoint(base_point, free_point, cont, contour_img, tcl_pred, centroid)
            if none_count > 3:
                break
            if ci is None:
                none_count += 1
                free_point = self.step_forward(free_point, h, w, stride, direction=1)
                continue

            none_count = 0
            ci_x, ci_y = ci
            if tcl_pred[ci_y, ci_x]:
                principal_curve_cw.append(np.array([ci_x, ci_y, radii_pred[ci_y, ci_x]]))

            # free point step forward
            free_point = self.step_forward(free_point, h, w, stride, direction=1)

        # counter-clockwise forward
        none_count = 0
        free_point = self.step_forward(free_point_backup, h, w, stride, direction=-1)
        while not is_in_same_axis(base_point, free_point):
            ci = self.find_center_innerpoint(base_point, free_point, cont, contour_img, tcl_pred, centroid)
            if none_count > 3:
                break
            if ci is None:
                none_count += 1
                free_point = self.step_forward(free_point, h, w, stride, direction=-1)
                continue

            none_count = 0
            ci_x, ci_y = ci
            if tcl_pred[ci_y, ci_x]:
                principal_curve_ccw.append(np.array([ci_x, ci_y, radii_pred[ci_y, ci_x]]))

            # free point step forward
            free_point = self.step_forward(free_point, h, w, stride, direction=-1)

        principal_curve = np.array([])
        if len(principal_curve_cw) > 0 and len(principal_curve_ccw) > 0:
            principal_curve = np.concatenate([principal_curve_ccw[::-1][::-1], principal_curve_cw])
        else:
            if not len(principal_curve_cw) > 0:
                principal_curve = np.array(principal_curve_ccw[::-1][::-1])
            if not len(principal_curve_ccw) > 0:
                principal_curve = np.array(principal_curve_cw)

        # print(principal_curve.shape)

        # self.debug_img *= 255
        # self.debug_img = self.debug_img[:, :, np.newaxis]
        # self.debug_img = cv2.cvtColor(self.debug_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # cv2.line(self.debug_img, (start[0], start[1]), (end[0], end[1]), (255, 0, 0), 1)
        # for x, y, r in principal_curve:
        #     cv2.circle(self.debug_img, (int(x), int(y)), 1, (0, 0, 255), -1)
        # cv2.imwrite('test_{}.jpg'.format(self.idx), self.debug_img)
        # self.debug_img = np.zeros((512, 512))
        # self.idx += 1

        return principal_curve

    def centerlize(self, x, y, tangent_cos, tangent_sin, mask, stride=1):
        """
        centralizing (x, y) using tangent line and normal line.
        :return:
        """
        # calculate normal sin and cos
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        # find upward
        _x, _y = x, y
        while self.is_inside(mask, _x, _y) and mask[int(_y), int(_x)]:
            _x = _x + normal_cos * stride
            _y = _y + normal_sin * stride
        end1 = np.array([_x, _y])

        # find downward
        _x, _y = x, y
        while self.is_inside(mask, _x, _y) and mask[int(_y), int(_x)]:
            _x = _x - normal_cos * stride
            _y = _y - normal_sin * stride
        end2 = np.array([_x, _y])

        # centralizing
        center = (end1 + end2) / 2

        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_mask, init_xy, direct=1):
        """
        Iteratively find center line in tcl mask using initial point (x, y)
        :param pred_sin: predict sin map
        :param pred_cos: predict cos map
        :param tcl_mask: predict tcl mask
        :param init_xy: initial (x, y)
        :param direct: direction [-1|1]
        :return:
        """

        x_init, y_init = init_xy

        sin = pred_sin[int(y_init), int(x_init)]
        cos = pred_cos[int(y_init), int(x_init)]
        radii = pred_radii[int(y_init), int(x_init)]

        x_shift, y_shift = self.centerlize(x_init, y_init, cos, sin, tcl_mask)
        result = []

        cycle_count = 0
        while self.is_inside(tcl_mask, x_shift, y_shift) and tcl_mask[int(y_shift), int(x_shift)]:
            cycle_count += 1
            if cycle_count > 100:
                break
            result.append(np.array([x_shift, y_shift, radii]))
            x, y = x_shift, y_shift

            sin = pred_sin[int(y), int(x)]
            cos = pred_cos[int(y), int(x)]

            x_c, y_c = self.centerlize(x, y, cos, sin, tcl_mask)

            sin_c = pred_sin[int(y_c), int(x_c)]
            cos_c = pred_cos[int(y_c), int(x_c)]
            radii = pred_radii[int(y_c), int(x_c)]

            # shift stride = +/- 0.5 * [sin|cos](theta)
            t = 0.5 * radii
            x_shift_pos = x_c + cos_c * t * direct  # positive direction
            y_shift_pos = y_c + sin_c * t * direct  # positive direction
            x_shift_neg = x_c - cos_c * t * direct  # negative direction
            y_shift_neg = y_c - sin_c * t * direct  # negative direction

            # if first point, select positive direction shift
            if len(result) == 1:
                x_shift, y_shift = x_shift_pos, y_shift_pos
            else:
                # else select point further by second last point
                dist_pos = norm2(result[-2][:2] - (x_shift_pos, y_shift_pos))
                dist_neg = norm2(result[-2][:2] - (x_shift_neg, y_shift_neg))
                if dist_pos > dist_neg:
                    x_shift, y_shift = x_shift_pos, y_shift_pos
                else:
                    x_shift, y_shift = x_shift_neg, y_shift_neg

        return result

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Find TCL's center points and radii of each point
        :param tcl_pred: output tcl mask, (512, 512)
        :param sin_pred: output sin map, (512, 512)
        :param cos_pred: output cos map, (512, 512)
        :param radii_pred: output radii map, (512, 512)
        :return: (list), tcl array: (n, 3) 3 denote (x, y, radii)
        """
        all_tcls = []

        conts, _ = cv2.findContours(tcl_pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in conts:
            # find an inner point of polygon
            init = self.find_innerpoint(cont)

            if init is None:
                continue

            x_init, y_init = init

            # find left tcl
            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init))
            tcl_left = np.array(tcl_left)
            # find right tcl
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init), direct=-1)
            tcl_right = np.array(tcl_right)
            # concat
            tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
            all_tcls.append(tcl)

        return all_tcls

    def is_real_crosspoint_pair(self, point1, point2, center, tcl_mask, mode='x'):
        if mode == 'x' and (point1[1]-center[1])*(point2[1]-center[1]) >= 0:
            return False
        elif mode == 'y' and (point1[0]-center[0])*(point2[0]-center[0]) >= 0:
            return False
        elif not tcl_mask[int((point1[1]+point2[1])/2), int((point1[0]+point2[0])/2)]:
            return False
        else:
            return True

    def mask_to_tcl_new(self, tcl_mask, tcl_cont, pred_radii, init_xy, direct=1):
        '''
        :param tcl_mask:
        :param tcl_cont:
        :param pred_radii:
        :param init_xy:
        :param direct:
        :return:
        '''
        x_init, y_init = init_xy
        cont_x = tcl_cont[:, 0, 0]
        cont_y = tcl_cont[:, 0, 1]

        # from x_init find nearest cont_x
        idx_x = np.argsort(np.abs(cont_x - x_init))[:5]

        crosspoints_x_idx = []
        for i, pre_idx in enumerate(idx_x[:-1]):
            for post_idx in idx_x[i+1:]:
                point1 = tcl_cont[pre_idx, 0, :]
                point2 = tcl_cont[post_idx, 0, :]
                if self.is_real_crosspoint_pair(point1, point2, init_xy, tcl_mask):
                    crosspoints_x_idx.append(pre_idx)
                    crosspoints_x_idx.append(post_idx)
                    break
            if len(crosspoints_x_idx) > 0:
                break

        # from y_init find nearest cont_y
        idx_y = np.argsort(np.abs(cont_y - y_init))[:5]

        crosspoints_y_idx = []
        for i, pre_idx in enumerate(idx_y[:-1]):
            for post_idx in idx_y[i+1:]:
                point1 = tcl_cont[pre_idx, 0, :]
                point2 = tcl_cont[post_idx, 0, :]
                if self.is_real_crosspoint_pair(point1, point2, init_xy, tcl_mask, mode='y'):
                    crosspoints_y_idx.append(pre_idx)
                    crosspoints_y_idx.append(post_idx)
                    break
            if len(crosspoints_y_idx) > 0:
                break

        init_xy = np.array(init_xy)
        _, _, w, h = cv2.boundingRect(tcl_cont)
        if len(crosspoints_x_idx) > 0 and not len(crosspoints_y_idx) > 0:
            crosspoints_idx = np.sort(crosspoints_x_idx)
        elif (not len(crosspoints_x_idx) > 0) and len(crosspoints_y_idx) > 0:
            crosspoints_idx = np.sort(crosspoints_y_idx)
        elif (not len(crosspoints_x_idx) > 0) and (not len(crosspoints_y_idx) > 0):
            return []
        elif np.linalg.norm(tcl_cont[crosspoints_x_idx[0], 0, :]-init_xy) + \
                np.linalg.norm(init_xy-tcl_cont[crosspoints_x_idx[1], 0, :]) < \
                np.linalg.norm(tcl_cont[crosspoints_y_idx[0], 0, :]-init_xy) + \
                np.linalg.norm(init_xy-tcl_cont[crosspoints_y_idx[1], 0, :]):
            crosspoints_idx = np.sort(crosspoints_x_idx)
        else:
            crosspoints_idx = np.sort(crosspoints_y_idx)

        if w >= h and len(crosspoints_x_idx) > 0:
            crosspoints_idx = np.sort(crosspoints_x_idx)
        elif w < h and len(crosspoints_y_idx) > 0:
            crosspoints_idx = np.sort(crosspoints_y_idx)

        # init
        idx1, idx2 = crosspoints_idx[0], crosspoints_idx[1]
        point1 = tcl_cont[idx1, 0, :]
        point2 = tcl_cont[idx2, 0, :]
        x_shift, y_shift = int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)

        stride = 2
        result = []
        while np.abs(idx1-idx2) > stride:
            if tcl_mask[int(y_shift), int(x_shift)]:
                result.append(np.array([x_shift, y_shift, pred_radii[y_shift, x_shift]]))

            # next step
            idx1 = (idx1 - direct*stride) % len(tcl_cont)
            idx2 = (idx2 + direct*stride) % len(tcl_cont)

            point1 = tcl_cont[idx1, 0, :]
            point2 = tcl_cont[idx2, 0, :]
            x_shift, y_shift = int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)

        return result

    def build_tcl_new(self, tcl_per_mask, tcl_per_cont, radii_pred):
        '''
        :param tcl_per_mask:
        :param tcl_per_cont:
        :param radii_pred:
        :return:
        '''
        all_tcls = []

        # find an inner point of polygon
        init = self.find_innerpoint(tcl_per_cont)

        if init is None or not tcl_per_mask[int(init[1]), int(init[0])]:
            return []

        x_init, y_init = init

        # find left tcl
        tcl_left = self.mask_to_tcl_new(tcl_per_mask, tcl_per_cont, radii_pred, (int(x_init), int(y_init)))
        tcl_left = np.array(tcl_left)
        # find right tcl
        tcl_right = self.mask_to_tcl_new(tcl_per_mask, tcl_per_cont, radii_pred, (int(x_init), int(y_init)), direct=-1)
        tcl_right = np.array(tcl_right)
        # concat
        tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
        all_tcls.append(tcl)

        return all_tcls

    def instance_detect(self, tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred, radii_pred):
        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        # find tcl in each predicted mask
        tcl_result = []
        conts, _ = cv2.findContours(tcl_pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in conts:
            drawing = np.zeros(tcl_pred_mask.shape, np.uint8)
            tcl_per_mask = cv2.fillPoly(drawing, [cont.astype(np.int32)], 1)

            # find disjoint regions
            tcl_per_mask = fill_hole(tcl_per_mask)

            # remove small regions
            tcl_per_conts, _ = cv2.findContours(tcl_per_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_area = cv2.contourArea(tcl_per_conts[0])
            if region_area < 50:
                continue
            # remove non-text-like regions
            _, (w, h), _ = cv2.minAreaRect(tcl_per_conts[0])
            if float(max(w, h)) / min(w, h) < 3 and region_area < 300:
                continue

            # slightly enlarge for easier to get tcl
            kernel = np.ones((5, 5), np.uint8)
            tcl_per_mask = cv2.dilate(tcl_per_mask, kernel, iterations=2)

            tcl = self.build_tcl(tcl_per_mask, sin_pred, cos_pred, radii_pred)
            if len(tcl) > 0:
                tcl_result.append(tcl)

        return tcl_result

    def full_detect(self, tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred, radii_pred):
        # find tcl in each predicted mask
        tcl_result = []
        conts, _ = cv2.findContours(tcl_pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for tcl_per_cont in conts:
            # remove small regions
            region_area = cv2.contourArea(tcl_per_cont)
            if region_area < 50:
                continue
            # remove non-text-like regions
            _, (w, h), _ = cv2.minAreaRect(tcl_per_cont)
            if float(max(w, h)) / min(w, h) < 3 and region_area < 150:
                continue

            drawing = np.zeros(tcl_pred_mask.shape, np.int8)
            tcl_per_mask = cv2.fillPoly(drawing, [tcl_per_cont.astype(np.int32)], 1)

            # slightly enlarge for easier to get tcl
            kernel = np.ones((5, 5), np.uint8)
            tcl_per_mask = cv2.dilate(tcl_per_mask.astype(np.uint8), kernel, iterations=2)

            tcl = self.build_tcl_new(tcl_per_mask, tcl_per_cont, radii_pred)
            if len(tcl) > 0:
                tcl_result.append(tcl)

        return tcl_result

    def complete_detect(self, tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred, radii_pred):
        # find tcl in each predicted mask
        tcl_result = []
        conts, _ = cv2.findContours(tcl_pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in conts:
            drawing = np.zeros(tcl_pred_mask.shape, np.uint8)
            tcl_per_mask = cv2.fillPoly(drawing, [cont.astype(np.int32)], 1)

            # find disjoint regions
            tcl_per_mask = fill_hole(tcl_per_mask)

            # remove small regions
            tcl_per_conts, _ = cv2.findContours(tcl_per_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_area = cv2.contourArea(tcl_per_conts[0])
            if region_area < 50:
                continue
            # remove non-text-like regions
            _, (w, h), _ = cv2.minAreaRect(tcl_per_conts[0])
            if float(max(w, h)) / min(w, h) < 3 and region_area < 300:
                continue

            nonzero_xy = np.transpose(np.nonzero(tcl_per_mask))[:, ::-1]
            if len(nonzero_xy) > 200:
                nonzero_xy = random.sample(nonzero_xy.tolist(), 200)
            else:
                nonzero_xy = nonzero_xy.tolist()

            nonzero_xyr = [[x, y, radii_pred[y, x]] for x, y in nonzero_xy]
            if len(nonzero_xyr) > 0:
                tcl = [np.array(nonzero_xyr)]
                tcl_result.append(tcl)

        return tcl_result