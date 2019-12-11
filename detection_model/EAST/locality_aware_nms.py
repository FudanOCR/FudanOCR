import numpy as np
from shapely.geometry import Polygon


def cal_distance(point1, point2):
    dis = np.sqrt(np.sum(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1])))
    return dis


# 基于海伦公式计算不规则四边形的面积
def helen_formula(coord):
    coord = np.array(coord).reshape((4, 2))
    # 计算各边的欧式距离
    dis_01 = cal_distance(coord[0], coord[1])
    dis_12 = cal_distance(coord[1], coord[2])
    dis_23 = cal_distance(coord[2], coord[3])
    dis_31 = cal_distance(coord[3], coord[1])
    dis_13 = cal_distance(coord[0], coord[3])
    p1 = (dis_01 + dis_12 + dis_13) * 0.5
    p2 = (dis_23 + dis_31 + dis_13) * 0.5
    # 计算两个三角形的面积
    area1 = np.sqrt(p1 * (p1 - dis_01) * (p1 - dis_12) * (p1 - dis_13))
    area2 = np.sqrt(p2 * (p2 - dis_23) * (p2 - dis_31) * (p2 - dis_13))
    return area1 + area2


def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    #g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    area_g = helen_formula(g[:8])
    area_p = helen_formula(p[:8])
    if area_g < area_p:
        g[:8] = (g[8] * g[:8] + 5 * p[8] * p[:8])/(g[8] + 5 * p[8])
    else:
        g[:8] = (5 * g[8] * g[:8] + p[8] * p[:8])/(5 * g[8] + p[8])
        #g[:8] = p[:8]
    g[8] = (g[8] + p[8])
    return g



def merge(g, p):
    res_box = np.zeros(9)
    # select higher score
    # res_box[8] = g[8] if g[8] > p[8] else p[8]
    res_box[8] = g[8] + p[8]
    # merge
    minx = min(g[0], p[0])
    maxx = max(g[4], p[4])
    miny = g[1] if g[8] > p[8] else p[1]
    maxy = g[5] if g[8] > p[8] else p[5]
    res_box[:8] = [minx, miny, maxx, miny, maxx, maxy, minx, maxy]

    return res_box


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def two_criterion_nms(S, thres):
    S = S[np.argsort(S[:, 8])][::-1]    # order by score (higher better)
    order = np.argsort(S[:, 9])[::-1]   # order by rescore (higher better)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def merge_nms(polys, thres):
    '''
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
