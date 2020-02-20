import os
import numpy as np
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

import model.detection_model.AdvancedEAST.config as cfg


class Anno:
    def __init__(self, img_list, img_dir, split='train'):
        '''Load annotations. Create one if not exist.'''
        self.img_list = img_list
        self.img_dir = img_dir
        self.split = split
        if not os.path.exists(cfg.cache_dir):
            os.mkdir(cfg.cache_dir)

        self.anno_dir = os.path.join(cfg.cache_dir, cfg.task_id + '/', split + '/gt/')
        self.nimg_dir = os.path.join(cfg.cache_dir, cfg.task_id + '/', split + '/img/')
        if not os.path.exists(self.anno_dir):
            os.makedirs(self.anno_dir)
        if not os.path.exists(self.nimg_dir):
            os.makedirs(self.nimg_dir)

    def check_anno(self):
        if os.path.isdir(self.anno_dir) and len(os.listdir(self.anno_dir)) == len(self.img_list) and os.path.isdir(self.nimg_dir) and len(os.listdir(self.nimg_dir)) == len(self.img_list):
            pass
        else:  # accelerate via multiprocessing
            pool = Pool(processes=cfg.num_process)
            with tqdm(total=len(self.img_list)) as pbar:
                for _, _ in tqdm(enumerate(pool.imap_unordered(self.create_anno, self.img_list))):
                    pbar.update()
            pool.close()
            pool.join()

    def create_anno(self, img_name):
        anno_file = os.path.join(self.anno_dir, img_name[:-4] + '.npy')
        nimg_file = os.path.join(self.nimg_dir, img_name[:-4] + '.jpg')
        if os.path.exists(anno_file) and os.path.exists(nimg_file):
            return
        with Image.open(os.path.join(self.img_dir, img_name)) as im:
            tsize = cfg.train_size
            dsize = np.asarray(im.size, dtype=np.float32)
            ratio = tsize / np.max(dsize)
            dsize = np.rint(dsize * ratio).astype(np.int32)
            scale_ratio_w = dsize[0] / im.width
            scale_ratio_h = dsize[1] / im.height
            delta_w = (tsize - dsize[0]) // 2
            delta_h = (tsize - dsize[1]) // 2

            # resize and save input image to cache_dir
            im = im.resize(dsize, Image.BICUBIC)
            new_img = Image.new("RGB", (tsize, tsize), (128, 128, 128))
            new_img.paste(im, ((tsize - dsize[0]) // 2, (tsize - dsize[1]) // 2))
            new_img.save(nimg_file, quality=95)

            if os.path.exists(anno_file):
                return
            with open(os.path.join(self.img_dir[:-4] + 'gt/', 'gt_' + img_name[:-4] + '.txt'), 'r', encoding='UTF-8-sig') as f:
                anno_list = f.readlines()

            gt = np.zeros((tsize // cfg.pixel_size, tsize // cfg.pixel_size, 7), dtype=np.float32)
            for i, anno in enumerate(anno_list):
                anno_column = anno.strip().split(',')
                anno_array = np.asarray(anno_column)
                xy_list = np.reshape(anno_array[:8].astype(np.float32), (4, 2))

                # rescale and reorder annatation
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w + delta_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h + delta_h
                xy_list = reorder_vertexes(xy_list)

                '''Prepare ground-truth map.'''
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)

                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(np.int32) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(np.int32) + 3
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(tsize // cfg.pixel_size, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(tsize // cfg.pixel_size, ji_max[0])
                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        # Genereate ground-truth
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                            # inside score
                            gt[i, j, 0] = 1
                            ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                # side-vertex code
                                gt[i, j, 1] = 1
                                gt[i, j, 2:3] = ith
                                # side-vertex geo
                                gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]
        np.save(anno_file, gt)

    def get_dir(self):
        '''Return anno_dir, nimg_dir.'''
        return self.anno_dir, self.nimg_dir


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    '''
    Part of shrinking process, for ground-truth preparation.

    by https://github.com/huoyijie/AdvancedEAST
    '''
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    '''
    Part of shrinking process, for ground-truth preparation.

    by https://github.com/huoyijie/AdvancedEAST
    '''
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


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
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)

    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]

    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0

    for index, i in zip(others, range(len(others))):

        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index

    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]

    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)

    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y

    return reorder_xy_list


def shrink(xy_list, ratio=cfg.shrink_ratio):
    '''
    Shrink, for ground-truth preparation.

    by https://github.com/huoyijie/AdvancedEAST
    '''
    if ratio == 0.0:
        return xy_list, xy_list

    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))

    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge

    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]

    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += cfg.epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])

    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)

    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)

    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=cfg.shrink_ratio):
    '''
    Shrink edge, for ground-truth preparation.

    by https://github.com/huoyijie/AdvancedEAST
    '''
    if ratio == 0.0:
        return

    start_point = edge
    end_point = (edge + 1) % 4

    long_start_sign_x = np.sign(xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = xy_list[start_point, 0] + long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])

    long_start_sign_y = np.sign(xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = xy_list[start_point, 1] + long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])

    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = xy_list[end_point, 0] + long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])

    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = xy_list[end_point, 1] + long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])
