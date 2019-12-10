import numpy as np
cimport numpy as np
from config import pixel_size, epsilon, side_vertex_pixel_threshold, trunc_threshold

from libcpp.deque cimport deque
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from cython.operator cimport dereference, preincrement

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef int PIXEL_SIZE = pixel_size
cdef float EPSILON = epsilon
cdef float SIDE_THRES = side_vertex_pixel_threshold
cdef float TRUNC_THRES = trunc_threshold

cdef bint set_intersection(cset[int]& a1, cset[int]& a2):
    cdef cset[int].iterator first1 = a1.begin()
    cdef cset[int].iterator first2 = a2.begin()
    cdef cset[int].iterator last1 = a1.end()
    cdef cset[int].iterator last2 = a2.end()
    while first1 != last1 and first2 != last2:
        if dereference(first1) < dereference(first2):
            preincrement(first1)
        else:
            if not dereference(first2) < dereference(first1):
                return True
            preincrement(first2)
    return False


cdef vector[int] create_vec(int m):
    cdef vector[int] tmp
    assert tmp.empty()
    tmp.push_back(m)
    return tmp


cdef vector[vector[int]] region_group(vector[cset[int]]& region_list):
    cdef deque[int] S
    assert S.empty()
    for i in range(region_list.size()):
        S.push_back(i)
    cdef vector[vector[int]] D
    assert D.empty()
    while S.size() > 0:
        m = S.front()
        S.pop_front()
        if S.size() == 0:
            D.push_back(create_vec(m))
        else:
            D.push_back(rec_region_merge(region_list, m, S))
    return D


cdef vector[int] rec_region_merge(vector[cset[int]]& region_list, int m, deque[int]& S):
    cdef vector[int] rows = create_vec(m)
    cdef vector[int] tmp
    assert tmp.empty()
    for i in range(S.size()):
        n = S[i]
        r1 = region_neighbor(region_list[m])
        r2 = region_neighbor(region_list[n])
        if set_intersection(r1, region_list[n]) or set_intersection(r2, region_list[m]):
            # 若m与n相交
            tmp.push_back(n)
    cdef deque[int].iterator dit = S.begin()
    for d in tmp:
        while dit != S.end():
            if dereference(dit) == d:
                dit = S.erase(dit)
                break
            else:
                preincrement(dit)
    cdef vector[int] rec
    assert rec.empty()
    for e in tmp:
        rec = rec_region_merge(region_list, e, S)
        for i in range(rec.size()):
            rows.push_back(rec[i])
    return rows


cdef cset[int] region_neighbor(cset[int]& region_set):
    cdef cset[int] neighbor
    assert neighbor.empty()
    cdef int i_min = 999
    cdef int j_min = 999
    cdef int j_max = 0
    cdef int ij, i, j
    cdef cset[int].iterator it = region_set.begin()
    while it != region_set.end():
        ij = dereference(it)
        i = int(ij / 1000)
        j = ij % 1000
        if i < i_min:
            i_min = i
        if j < j_min:
            j_min = j
        if j > j_max:
            j_max = j
        neighbor.insert((i + 1) * 1000 + j)
        preincrement(it)
    neighbor.insert((i_min + 1) * 1000 + j_min - 1)
    neighbor.insert((i_min + 1) * 1000 + j_min + 1)
    return neighbor


cdef cset[int] create_set(int pair):
    cdef cset[int] tmp
    assert tmp.empty()
    tmp.insert(pair)
    return tmp


cdef vector[float] create_vec_list(int length):
    cdef vector[float] tmp_list
    assert tmp_list.empty()
    for k in range(length):
        tmp_list.push_back(0.0)
    return tmp_list


def nms(np.ndarray[DTYPE_t, ndim=3] predict, int[:] active_x, int[:] active_y):
    cdef vector[cset[int]] region_list
    assert region_list.empty()
    cdef unsigned int x_max = active_x.shape[0]
    cdef bint merge
    cdef cset[int] neighbor, region_set
    cdef int ij, i, j
    for idx in range(x_max):
        i = active_x[idx]
        j = active_y[idx]
        merge = False
        for k in range(region_list.size()):
            neighbor = create_set(i * 1000 + j - 1)
            if set_intersection(region_list[k], neighbor):
                region_list[k].insert(i * 1000 + j)
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
        if not merge:
            region_set = create_set(i * 1000 + j)
            region_list.push_back(region_set)

    cdef vector[vector[int]] D = region_group(region_list)
    cdef int D_len = D.size()
    cdef vector[vector[float]] quad_list
    cdef vector[vector[float]] score_list
    assert quad_list.empty() and score_list.empty()
    cdef float total_score[8]
    cdef float pv[4]
    cdef float score, ith_score, px, py
    cdef cset[int].iterator it
    cdef int ith
    for g_th in range(D_len):
        group = D[g_th]
        quad_list.push_back(create_vec_list(8))
        score_list.push_back(create_vec_list(4))
        for k in range(8):
            total_score[k] = 0.0
        for row_id in range(group.size()):
            row = group[row_id]
            it = region_list[row].begin()
            while it != region_list[row].end():
                ij = dereference(it)
                i = int(ij / 1000)
                j = ij % 1000
                score = predict[i, j, 1]
                if score >= SIDE_THRES:
                    ith_score = predict[i, j, 2]
                    if not (TRUNC_THRES <= ith_score < 1 - TRUNC_THRES):
                        ith = 0 if ith_score < TRUNC_THRES else 1
                        # total_score[ith * 2:(ith + 1) * 2] += score
                        for k in range(4):
                            total_score[ith * 4 + k] += score
                        px = (j + 0.5) * PIXEL_SIZE
                        py = (i + 0.5) * PIXEL_SIZE
                        # p_v = [px, py] + np.reshape(predict[i, j, 3:7], (2, 2))
                        pv[0] = predict[i, j, 3] + px
                        pv[1] = predict[i, j, 4] + py
                        pv[2] = predict[i, j, 5] + px
                        pv[3] = predict[i, j, 6] + py
                        # quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
                        for k in range(4):
                            quad_list[g_th][ith * 4 + k] += pv[k] * score
                preincrement(it)
        # score_list[g_th] = total_score[:, 0]
        for k in range(4):
            score_list[g_th][k] = total_score[k * 2]
        # quad_list[g_th] /= (total_score + EPSILON)
        for k in range(8):
            quad_list[g_th][k] /= (total_score[k] + EPSILON)
    np_score_list = np.asarray(score_list, dtype=DTYPE)
    np_quad_list = np.asarray(quad_list, dtype=DTYPE).reshape(D_len, 4, 2)
    return np_score_list, np_quad_list
