 # dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
from yacs.config import CfgNode as CN
from main import opt

def read_config_file(config_file):
 f = open(config_file)
 opt = CN.load_cfg(f)
 return opt


args = read_config_file(opt.config_file)

ic15_root_dir = args.dataset_root_dir
ic15_train_data_dir = ic15_root_dir + '/train_data/'
ic15_train_gt_dir = ic15_root_dir + '/train_groundTruth/'
ic15_test_data_dir = ic15_root_dir + '/test_data/'
ic15_test_gt_dir = ic15_root_dir + '/test_groundTruth/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')
        if gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)
        box = [int(gt[i]) for i in range(8)]
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(box)
    return np.array(bboxes), tags


def get_bboxes_LSVT(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')

        # print(len(gt))
        if len(gt) != 9:
            for i in range(10,len(gt)):
                gt[9] += gt[i]
            gt = gt[:9]

        if gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)

        num_points = len(gt)-1 #total points for boxes
        #print num_points
        # print("Debug")
        # for element in gt:
        #     print(element)
        # print("Debug Finish")

        box = [int(gt[i]) for i in range(num_points)]
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * (num_points/2))
        bboxes.append(box)


    return bboxes, tags

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if len(shrinked_bbox) <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        #print 'bbox3',shrinked_bbox
        shrinked_bboxes.append(shrinked_bbox)
    
    return shrinked_bboxes

class IC15Loader(data.Dataset):
    def __init__(self, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        data_dirs = [ic15_train_data_dir]
        gt_dirs = [ic15_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                
                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes = []
        bboxes, tags = get_bboxes_LSVT(img, gt_path)
        
        if self.is_transform:
            img = random_scale(img, self.img_size[0])

        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        #print 'boxes:',len(bboxes)
        boxes = [] #store the final boxes
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                num_points = len(bboxes[i])
                box = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (num_points/2)), (num_points / 2, 2)).astype('int32')
            
                cv2.drawContours(gt_text, [box], -1, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [box], -1, 0, -1)
                boxes.append(box)

        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(boxes, rate)
            for i in range(len(boxes)):
                ker = np.array(kernel_bboxes[i])
                cv2.drawContours(gt_kernel, [ker], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
        
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()
        # '''

        return img, gt_text, gt_kernels, training_mask