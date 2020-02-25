import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
from PIL import Image
from model.detection_model.TextSnake_pytorch.util.config import config as cfg
from skimage.draw import polygon as drawpoly
from model.detection_model.TextSnake_pytorch.util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin


class TextInstance(object):

    def __init__(self, points, orient, text, illegibility, language=None):
        self.orient = orient
        self.text = text
        if language is not None:
            self.language = language
        self.illegibility = illegibility
        self.points = np.array(points)

        # self.points = []
        # # remove point if area is almost unchanged after removing
        # ori_area = cv2.contourArea(points)
        # for p in range(len(points)):
        #     index = list(range(len(points)))
        #     index.remove(p)
        #     area = cv2.contourArea(points[index])
        #     if np.abs(ori_area - area) / ori_area > 0.017:
        #         self.points.append(points[p])
        # self.points = np.array(self.points)

    def find_bottom_and_sideline(self, points=None):
        if points is None:
            self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
            self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence
        else:
            self.points = points
            self.bottoms = find_bottom(points)  # find two bottoms of this Text
            self.e1, self.e2 = find_long_edges(points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        Args:
            n_disk: number of disks

        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def parse_mat(self, mat_path):
        """
        .mat file parser
        Args:
            mat_path: (str), mat file path

        Returns:
            (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4: # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        illegal_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#' or polygon.text == '###':
                cv2.fillPoly(illegal_mask, [polygon.points.astype(np.int32)], color=(1,))
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))

        return tr_mask, train_mask, illegal_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        Args:
            mask: input mask
            polygon: polygon to draw
            value: fill value

        Returns:

        """

        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(cfg.input_size, cfg.input_size))
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0], mask.shape[1]))
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius, \
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.2, shrink=5):

        # TODO: shrink 1/2 * radius at two line end
        while len(center_line)-1 - shrink < 0 or shrink > len(center_line):
            if shrink == 1:
                break
            print('shrink {} too large'.format(shrink))
            shrink -= 1
            shrink = shrink if shrink > 0 else 1

        for i in range(shrink, len(center_line) - 1 - shrink):
            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)

    def get_training_data(self, image, polygons, image_id, image_path, image_shape):

        H, W, _ = image_shape

        # if self.transform:
        #     image, polygons = self.transform(image, copy.copy(polygons))

        for i, polygon in enumerate(polygons):
            if polygon.text != '###' and polygon.text != '#':
                polygon.find_bottom_and_sideline(polygon.points)

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '###' and polygon.text != '#':
                if len(polygon.e1) > 0 and len(polygon.e2) > 0:
                    sideline1, sideline2, center_points, radius = polygon.disk_cover(n_disk=cfg.n_disk)
                    self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map)
        tr_mask, train_mask, illegal_mask = self.make_text_region(image, polygons)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W,
            'illegal_mask': illegal_mask
        }
        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta

    def __len__(self):
        raise NotImplementedError()