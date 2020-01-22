import os
import time
import argparse
import numpy as np
import torch
from multiprocessing import Pool, RLock, set_start_method
from PIL import Image, ImageDraw
from tqdm import tqdm
import json

from utils.preprocess import point_inside_of_quad
from utils.data_utils import transform
from network.AEast import East
from nms.nms import nms
from tools.Pascal_VOC import eval_func
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskid', '-t', required=True, type=str, help='task id')
    parser.add_argument('--draw', action='store_true', help='visualize and save as image')
    return parser.parse_args()


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def resize_image(img_size):
    '''size shall be divided by 32, required by network.'''
    dsize = np.asarray(img_size, dtype=np.float32)
    tsize = cfg.train_size
    ratio = tsize / np.max(dsize)
    dsize = np.rint(dsize * ratio).astype(np.int32)
    dsize = dsize - dsize % 32
    return dsize[0], dsize[1]


def res2json(result_dir):
    res_list = os.listdir(result_dir)
    res_dict = {}
    for rf in tqdm(res_list, desc='toJSON'):
        if rf[-4:] == '.txt':
            respath = os.path.join(result_dir, rf)
            with open(respath, 'r') as f:
                reslines = f.readlines()
            reskey = rf[:-4]
            res_dict[reskey] = [{'points': np.rint(np.asarray(l.replace('\n', '').split(','), np.float32)).astype(np.int32).reshape(-1, 2).tolist()} for l in reslines]

    jpath = os.path.join(result_dir, 'res.json')
    with open(jpath, 'w') as jf:
        json.dump(res_dict, jf)
    return jpath

# copy 上面的
def res2json_1(result_dir):
    res_list = os.listdir(result_dir)
    res_dict = {}
    for rf in tqdm(res_list, desc='toJSON'):
        if rf[-4:] == '.txt':
            respath = os.path.join(result_dir, rf)
            with open(respath, 'r') as f:
                reslines = f.readlines()
            reskey = rf[3:-4]
            res_dict[reskey] = [{'points': np.rint(np.asarray(l.replace('\n', '').split(',')[:8], np.float32)).astype(np.int32).reshape(-1, 2).tolist()} for l in reslines]

    jpath = os.path.join(result_dir, 'res.json')
    with open(jpath, 'w') as jf:
        json.dump(res_dict, jf)
    return jpath


class Wrapped:
    def __init__(self, model, img_dir, isDraw):
        self.model = model
        self.img_dir = img_dir
        self.isDraw = isDraw
        self.result_dir = cfg.result_dir + args.taskid + '/'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def init_lock(self, lock_):
        global lock
        lock = lock_

    def __call__(self):
        img_list = [img_name for img_name in os.listdir(self.img_dir)]
        miss = []

        if cfg.batch_size_per_gpu > 1:
            set_start_method('forkserver')
            lock_ = RLock()
            processes = 2
            pool = Pool(processes=processes, initializer=self.init_lock, initargs=(lock_,))
            with tqdm(total=len(img_list), desc='Detect') as pbar:
                for _, r in enumerate(pool.imap_unordered(self.process, img_list)):
                    if r[0] == 0:
                        miss.append(r[1])
                        # tqdm.write(f"{r[1]}: {r[0]} quads.")
                    pbar.update()
            pool.close()
            pool.join()
        else:
            for img_name in tqdm(img_list):
                r = self.process(img_name)
                if r[0] == 0:
                    miss.append(r[1])

        print(f"{len(miss)} images no detection.")
        print(miss)
        input_json_path = res2json(self.result_dir)
        gt_json_path = res2json_1("/home/msy/ICDAR15/Text_Localization/val/gt/")
        # gt_json_path = cfg.gt_json_path
        eval_func(input_json_path, gt_json_path, cfg.iou_threshold)

    def process(self, img_name):
        txt_path = self.result_dir + img_name[:-4] + '.txt'
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f_txt:
                txt_items = f_txt.readlines()
                return len(txt_items), img_name

        img_path = os.path.join(self.img_dir, img_name)
        im = Image.open(img_path).convert('RGB')
        if cfg.predict_cut_text_line:
            im_array = np.array(im, dtype=np.float32)

        d_width, d_height = resize_image(im.size)
        scale_ratio_w = d_width / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_width, d_height), Image.BICUBIC)

        x = transform(im)
        x = x[np.newaxis, :]
        # lock.acquire()
        y = self.model(x.cuda()).cpu().detach().numpy()
        # lock.release()

        y = np.squeeze(y)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], cfg.pixel_threshold)
        activation_pixels = np.asarray(np.where(cond), dtype=np.int32)

        quad_scores, quad_after_nms = nms(y, activation_pixels[0], activation_pixels[1])

        if self.isDraw:
            quad_im = im.copy()
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'aqua'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                          width=line_width, fill=line_color)
            im.save(self.result_dir + img_name[:-4] + '_act.jpg')

            quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        invalid = 0
        for score, geo, s in zip(quad_scores, quad_after_nms, range(len(quad_scores))):
            if np.amin(score) > 0:
                if self.isDraw:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='aqua')
                if cfg.predict_cut_text_line:
                    self.cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_name, s)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            else:
                invalid += 1
        if self.isDraw:
            quad_im.save(self.result_dir + img_name[:-4] + '_predict.jpg')

        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)
        return (len(txt_items), img_name)

    def cut_text_line(self, geo, scale_ratio_w, scale_ratio_h, im_array, img_name, s):
        geo /= [scale_ratio_w, scale_ratio_h]
        p_min = np.amin(geo, axis=0)
        p_max = np.amax(geo, axis=0)
        min_xy = p_min.astype(int)
        max_xy = p_max.astype(int) + 2
        sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
        for m in range(min_xy[1], max_xy[1]):
            for n in range(min_xy[0], max_xy[0]):
                if not point_inside_of_quad(n, m, geo, p_min, p_max):
                    sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
        sub_im = Image.fromarray(sub_im_arr.astype('uint8')).convert('RGB')
        sub_im.save(self.result_dir + img_name[:-4] + '_subim%d.jpg' % s)


if __name__ == '__main__':
    args = parse_args()

    print(f'Task id: {args.taskid}')
    assert int(args.taskid[2:]) in cfg.size_group, f'input size shall be in {cfg.size_group}'
    # cp_file = args.taskid + '_best.pth.tar'
    cp_file = '3T1280_best.pth.tar'
    cp_path = os.path.join(cfg.result_dir, cp_file)
    assert os.path.isfile(cp_path), 'Checkpoint file does not exist.'
    print(f'Loading {cp_path}')
    checkpoint = torch.load(cp_path)

    model = East()
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    wrap = Wrapped(model, cfg.val_img, args.draw)
    wrap()
