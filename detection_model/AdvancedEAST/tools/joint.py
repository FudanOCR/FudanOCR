import numpy as np
import cv2 as cv
from numpy.linalg import norm

from utils.preprocess import reorder_vertexes


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
        pts = np.asarray(pts.strip().split(','))
        pts = pts[:8].astype(np.float32).reshape(4, 2)
        pts = reorder_vertexes(pts)

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
                    v1_len = norm(v1)

                    q0 = plist[j][0][n][0]
                    q1 = plist[j][0][n][1]
                    v2 = q1 - q0
                    v2_len = norm(v2)

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
        xy = np.asarray(xy).reshape(-1).tolist()
        xy = ",".join(list(map(str, xy)))
        xylist.append(xy)

    return xylist


if __name__ == "__main__":
    curve = [[84, 1001],
             [125, 988],
             [165, 1013],
             [189, 1054],
             [174, 1075],
             [149, 1041],
             [118, 1024],
             [87, 1032]]
    pts_list = [
        '84,1001,125,988,118,1024,87,1032',
        '125,988,165,1013,149,1041,118,1024',
        '165,1013,189,1054,174,1075,149,1041'
    ]
    xylist = joint(pts_list)
    pts = np.asarray(xylist[0].strip().split(',')).astype(np.float32).reshape(-1, 2)
    pts = np.rint(pts).astype(np.int32)
    img = cv.imread('/home/ak/temp/gt_3.jpg')
    for i, pt in enumerate(pts):
        img = cv.polylines(img, (pt[0], pt[1]), 5, (255, i * 20, 0))
    cv.imshow('img', img)
    cv.waitKey(0)
