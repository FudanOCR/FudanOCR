import numpy as np
import cv2 as cv
from numpy.linalg import norm
import math

def reorder_vertexes(xy_list):
    '''
    Reorder vertexes.

    by https://github.com/huoyijie/AdvancedEAST
    '''
    reorder_xy_list = np.zeros_like(xy_list)

    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]

    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index

    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))

    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)

    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]

    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0

    for index, i in zip(others, range(len(others))):

        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index

    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]

    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (xy_list[second_v, 0] - xy_list[fourth_v, 0] + 1e-4)

    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y

    return reorder_xy_list

def joint(pts_list):
    """
    后处理拼接算法：
    reorder排序顶点，根据距离找短边，
    然后两个文本框的两组两个短边两两比较(共四次)，
    根据夹角阈值和中点连线距离阈值判断是否拼接，
    然后以短边中点为新的拼接点，且新拼接点不参与后续拼接。
    Argument:

    Input: A list of string of 4 points: "x1, y1, x2, y2, x3, y3, x4, y4"

    Output: A list of string of 4+ points.
    """
    plist = []
    for pts in pts_list:
        #pts = np.array(pts.strip().split(','))
        pts = pts[:8].astype(np.float32).reshape(4, 2)
        #print('pts1',pts)
        pts = reorder_vertexes(pts)
        #print('pts1', pts)

        min_side_len = np.inf
        min_side_num = 0
        for i in range(4):
            side_len = norm(pts[(i + 1) % 4] - pts[i])
            if side_len < min_side_len:
                min_side_len = side_len
                min_side_num = i
        pair0 = np.vstack([pts[min_side_num], pts[(min_side_num + 1) % 4]])
        pair1 = np.vstack([pts[(min_side_num + 3) % 4], pts[(min_side_num + 2) % 4]])
        # plist再细分为参与拼接和不参与拼接的两个子list
        plist.append([[pair0, pair1], []])

    #print(plist)
    i = 0
    plen = len(plist)
    while(i < plen):
        j = i + 1
        while(j < plen):
            # plist[i][0], plist[j][0]两个文本框的两组两个短边两两比较
            # NMS排除了两个框近乎重叠的情况，所以只要拼接就break
            for m in range(0, 2):
                judge = False
                for n in range(0, 2):
                    # 判断拼接
                    p0 = plist[i][0][m][0]
                    p1 = plist[i][0][m][1]
                    v1 = p1 - p0
                    v1_len = norm(v1)#第一个框的第m(0,1)条短边长度

                    q0 = plist[j][0][n][0]
                    q1 = plist[j][0][n][1]
                    v2 = q1 - q0
                    v2_len = norm(v2)#第二个框的第n(0,1)条短边长度

                    if not (1 / 1.2) < v1_len / v2_len < 1.2:
                        continue
                    cos_ = (v1 @ v2) / (v1_len * v2_len)
                    if cos_ < 0:
                        q0, q1 = q1, q0
                    if np.abs(cos_) < 1:
                        angle = np.arccos(cos_) * 180.0 / np.pi
                        if angle > 90:
                            angle = 180 - angle
                        if angle > 60:
                            continue
                    mid0 = (p0 + p1) / 2
                    mid1 = (q0 + q1) / 2
                    mid_len = norm(mid1 - mid0)
                    if mid_len > max(v1_len, v2_len) / 2:
                        continue
                    judge = True
                    if judge:
                        print("Joint!")
                        # 中点为拼接点
                        mid = [(p0 + q0) / 2, (p1 + q1) / 2]
                        plist[i][1].append(mid)
                        plist[i][0].pop(m)
                        plist[i][0].append(plist[j][0][1 - n])
                        plist.pop(j)
                        # 然后重新遍历
                        break
                if judge:
                    break
            j = i + 1 if judge else j + 1
            plen = len(plist)
        i += 1

    xylist = []
    for pts in plist:
        xy = pts[0]
        xy.extend(pts[1])
        xy = np.array(xy).reshape(-1).tolist()
        xy = ",".join(list(map(str, xy)))
        xylist.append(xy)

    return xylist


if __name__ == "__main__":
    curve = np.array([[84, 1001],
             [125, 988],
             [165, 1013],
             [189, 1054],
             [174, 1075],
             [149, 1041],
             [118, 1024],
             [87, 1032]],np.int32)
    curve = curve.reshape((-1,1,2))
    pts_list = [
        '84,1001,125,988,118,1024,87,1032',
        '125,988,165,1013,149,1041,118,1024',
        '165,1013,189,1054,174,1075,149,1041'
    ]
    xylist = joint(pts_list)
    pts = np.array(xylist[0].strip().split(',')).astype(np.float32).reshape(-1, 2)
    pts = np.rint(pts).astype(np.int)
    print(pts)
    pts.tolist()
    pts1 = []
    for i in range(len(pts)):
        pts[i] = tuple(pts[i])
        pts1.append(pts[i])
    print(pts1)
    # compute centroid
    cent = (sum([p[0] for p in pts1]) / len(pts1), sum([p[1] for p in pts1]) / len(pts1))
    print(cent)
    # sort by polar angle
    pts1.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

    for i in range(len(pts1)):
        pts1[i] = list(pts1[i])
    print(pts1)
    pts1 = np.array(pts1)
    pts1 = pts1.reshape((-1,1,2))
    #print(pts)
    img = cv.imread('/Users/leiboss/desktop/gt_3.jpg')
    #img = cv.polylines(img,[curve],True,color=(0, 0, 255),thickness=2)
    img = cv.polylines(img, [pts1], True, color=(0, 255, 255), thickness=2)
    cv.imwrite('/Users/leiboss/desktop/gt_3_test.jpg', img)


