import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg

from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir, write_result_ICDAR_MASKRRPN2polys
from PIL import Image
import time
import json
from tqdm import tqdm
from Pascal_VOC import eval_func
from link_boxes import merge
import pycocotools.mask as maskUtils
from skimage.measure import find_contours


def topoly(mask):
    # mask = maskUtils.decode(rle)
    #     print(maskUtils.area(rle[0]))
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    # print('masksum:', np.sum(mask), mask.shape)
    area = np.sum(mask).astype(np.float) # float(maskUtils.area(rle))
    if len(contours) == 0:
        return [[]], area
    #     print(contours)
    poly = np.fliplr(contours[0]).tolist()

    return poly, area


def get_mask(box,shape):
    """根据box获取对应的掩膜"""
    tmp_mask=np.zeros(shape,dtype="uint8")
    tmp=np.array(box,dtype=np.int32).reshape(-1,2)
    cv2.fillPoly(tmp_mask, [tmp], (255))
#     tmp_mask=cv2.bitwise_and(tmp_mask,mask)
    return tmp_mask,cv2.countNonZero(tmp_mask)


def comput_mmi(area_a,area_b,intersect):
    """
    计算MMI,2018.11.23 add
    :param mask_a: 实例文本a的mask的面积
    :param mask_b: 实例文本b的mask的面积
    :param intersect: 实例文本a和实例文本b的相交面积
    :return:
    """
    if area_a==0 or area_b==0:
        area_a+=EPS
        area_b+=EPS
        print("the area of text is 0")
    return max(float(intersect)/area_a,float(intersect)/area_b)


def mask_nms(dets, shape, thres=0.3,conf_thres=0.5):
    """
    mask nms 实现函数
    :param dets: 检测结果，[{'points':[[],[],[]],'confidence':int},{},{}]
    :param mask: 当前检测的mask
    :param thres: 检测的阈值
    """
    # 获取bbox及对应的score
    bbox_infos=[]
    areas=[]
    scores=[]
    for quyu in dets:
        if quyu['confidence']>conf_thres:
            bbox_infos.append(quyu['points'])
            areas.append(quyu['area'])
            scores.append(quyu['confidence'])
#     print('before ',len(bbox_infos))
    keep=[]
    order=np.array(scores).argsort()[::-1]
#     print("order:{}".format(order))
    nums=len(bbox_infos)
    suppressed=np.zeros((nums), dtype=np.int)
#     print("lens:{}".format(nums))

    # 循环遍历
    for i in range(nums):
        idx=order[i]
        if suppressed[idx]==1:
            continue
        keep.append(idx)
        mask_a,area_a=get_mask(bbox_infos[idx],shape)
        for j in range(i,nums):
            idx_j=order[j]
            if suppressed[idx_j]==1:
                continue
            mask_b, area_b = get_mask(bbox_infos[idx_j],shape)

            # 获取两个文本的相交面积
            merge_mask=cv2.bitwise_and(mask_a,mask_b)
            area_intersect=cv2.countNonZero(merge_mask)

            #计算MMI
            mmi=comput_mmi(area_a,area_b,area_intersect)
            # print("area_a:{},area_b:{},inte:{},mmi:{}".format(area_a,area_b,area_intersect,mmi))

            if mmi >= thres :
                suppressed[idx_j] = 1
#                 ormask=cv2.bitwise_or(mask_a,mask_b)
#                 sumarea=cv2.countNonZero(ormask)
#                 padded_mask = np.zeros((ormask.shape[0] + 2, ormask.shape[1] + 2), dtype=np.uint8)
#                 padded_mask[1:-1, 1:-1] = ormask
#                 contours = find_contours(padded_mask, 0.5)
#                 poly=np.fliplr(contours[0]).tolist()
#                 bbox_infos[idx]=poly
#                 areas[idx]=sumarea
    dets=[]
    for kk in keep:
        dets.append({
            'points': bbox_infos[kk],
            'confidence':scores[kk]
        })
    return dets


def res2json(result_dir):
    res_list = os.listdir(result_dir)

    res_dict = {}

    for rf in tqdm(res_list):
        if rf[-4:] == '.txt':
            respath = os.path.join(result_dir, rf)
            reslines = open(respath, 'r').readlines()
            reskey = rf[4:-4]
            polys = []
            for l in reslines:
                poly_pts = np.array(l.replace('\n', '').split(','), np.int).reshape(-1, 2)
                if poly_pts.shape[0] > 2:
                    polys.append({'points':poly_pts.tolist()})

            res_dict[reskey] = polys#[{'points':np.array(l.replace('\n', '').split(','), np.int).reshape(-1, 2).tolist()} for l in reslines]
            # print('res_dict[reskey]:', res_dict[reskey])

    json_tarf = os.path.join(result_dir, 'res.json')

    if os.path.isfile(json_tarf):
        print('Json file found, removing it...')
        os.remove(json_tarf)

    j_f = open(json_tarf, 'w')
    json.dump(res_dict, j_f)
    print('json dump done', json_tarf)

    return json_tarf

config_file = 'configs/Mask_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_MASK_RFPN_word_margin.yaml' #'#"configs/ICDAR2019_det_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_4scales_angle_norm.yaml" #e2e_rrpn_R_50_C4_1x_ICDAR13_15_trial_test.yaml

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.freeze()
# cfg.MODEL.WEIGHT = 'models/IC-13-15-17-Trial/model_0155000.pth'

vis = True
merge_box = cfg.TEST.MERGE_BOX
result_dir = os.path.join('results', config_file.split('/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

if merge_box:
    result_dir += '_merge_box'

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


coco_demo = RRPNDemo(
    cfg,
    min_image_size=896,
    confidence_threshold=0.87,
)

dataset_name = cfg.TEST.DATASET_NAME

testing_dataset = {
    'LSVT': {
        'testing_image_dir': '../datasets/LSVT/train_full_images_0/train_full_images_0/',
        'off': [0, 3000]
    },
    'ArT': {
        'testing_image_dir': '../datasets/ArT/ArT_detect_train/train_images',
        'off': [4000, 5603]
    },
}

image_dir = testing_dataset[dataset_name]['testing_image_dir']
# vocab_dir = testing_dataset[dataset_name]['test_vocal_dir']
off_group = testing_dataset[dataset_name]['off']
# load image and then run prediction
# image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
# imlist = os.listdir(image_dir)[off_group[0]:off_group[1]]



print('************* META INFO ***************')
print('config_file:', config_file)
print('result_dir:', result_dir)
print('image_dir:', image_dir)
print('weights:', cfg.MODEL.WEIGHT)
print('merge_box:', merge_box)
print('***************************************')

thres=0.8              #mask nms的阈值，大于thres的文本区域剔除
conf_thres=0.4

#num_images = len(imlist)
cnt = 0
num_images = off_group[1] - off_group[0]

for idx in range(off_group[0], off_group[1]):
    image = 'gt_' + str(idx) + '.jpg'
    impath = os.path.join(image_dir, image)
    # print('image:', impath)
    img = cv2.imread(impath)
    cnt += 1
    tic = time.time()
    predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
    toc = time.time()

    print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
    bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

    mask_list = bounding_boxes.get_field('mask')
    score_list = bounding_boxes.get_field('scores')

    if merge_box:
        bboxes_np_reverse = bboxes_np.copy()
        bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
        bboxes_np_reverse = merge(bboxes_np_reverse)
        bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
        bboxes_np = bboxes_np_reverse

    width, height = bounding_boxes.size

    if vis:
        # predictions.show()
        # print('mask_list:', len(mask_list), mask_list[0].shape)
        mask_total = np.zeros(img.shape[:2])
        for idx in range(len(mask_list)):
            # print('box_list:', bboxes_np[idx])
            mask = mask_list[idx]
            mask_np = mask.data.cpu().numpy()
            mask_total += mask_np[0] * 255

        mask_im = Image.fromarray(mask_total.astype(np.uint8))
        scale = 768.0 / mask_im.size[0]
        scaled_size = (int(scale * mask_im.size[0]), int(scale * mask_im.size[1]))
        mask_im = mask_im.resize(scaled_size)
        #mask_im.show()
        mask_im.save('re_img/mask_' + image, 'jpeg')
        pil_image = vis_image(Image.fromarray(img), bboxes_np)
        pil_image = pil_image.resize(scaled_size)
        pil_image.save('re_img/box_' + image, 'jpeg')
        # time.sleep(20)
    #  else:
    poly_list = []
    # mask_total = np.zeros(mask_list[0].shape[1:])

    res_list = []

    for idx in range(len(mask_list)):
        # print('box_list:', bboxes_np[idx])
        mask = mask_list[idx]
        mask_np = mask.data.cpu().numpy()[0]
        score = score_list[idx].data.cpu().numpy()
        # print('mask_np', mask_np.shape, np.unique(mask_np))
        contours = cv2.findContours(((mask_np > 0) * 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print('mask_np:', np.unique(mask_np), contours)
        if len(contours[1]) > 0:
            poly_list.append(contours[1][0].reshape(-1, 2))

        '''
        poly, area = topoly(mask_np)

        res_list.append({
            'points': poly,
            'confidence': score,
            'area': area,
            'size': mask_np.shape
        })
        '''
    # res_list = mask_nms(res_list, res_list[0]['size'], thres, conf_thres)
    # for res in res_list:
    #     poly_list.append(np.array(res['points']).reshape(-1, 2))

    write_result_ICDAR_MASKRRPN2polys(image[:-4], poly_list, threshold=0.7, result_dir=result_dir, height=height, width=width)
    #im_file, dets, threshold, result_dir, height, width
    #cv2.imshow('win', predictions)
    #cv2.waitKey(0)


if dataset_name == 'IC15':
    zipfilename = os.path.join(result_dir, 'submit_' + config_file.split('/')[-1].split('.')[0] + '_' + cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0] + '.zip')
    if os.path.isfile(zipfilename):
        print('Zip file exists, removing it...')
        os.remove(zipfilename)
    zip_dir(result_dir, zipfilename)
    comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
    # print(comm)
    print(os.popen(comm, 'r'))
elif dataset_name == 'LSVT':
    # input_json_path = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0190000/res.json'
    gt_json_path = '../datasets/LSVT/train_full_labels.json'
    # to json
    input_json_path = res2json(result_dir)
    eval_func(input_json_path, gt_json_path)
elif dataset_name == 'ArT':
    # input_json_path = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0190000/res.json'
    gt_json_path = '../datasets/ArT/ArT_detect_train/train_labels.json'
    # to json
    input_json_path = res2json(result_dir)
    eval_func(input_json_path, gt_json_path)
else:
    pass
