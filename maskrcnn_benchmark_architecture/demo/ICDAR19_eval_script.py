import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir
from PIL import Image
import time
import json
from tqdm import tqdm
from Pascal_VOC import eval_func
from link_boxes import merge

def res2json(result_dir):
    res_list = os.listdir(result_dir)

    res_dict = {}

    for rf in tqdm(res_list):
        if rf[-4:] == '.txt':
            respath = os.path.join(result_dir, rf)
            reslines = open(respath, 'r').readlines()
            reskey = rf[4:-4]
            res_dict[reskey] = [{'points':np.array(l.replace('\n', '').split(','), np.int).reshape(-1, 2).tolist()} for l in reslines]

    json_tarf = os.path.join(result_dir, 'res.json')

    if os.path.isfile(json_tarf):
        print('Json file found, removing it...')
        os.remove(json_tarf)

    j_f = open(json_tarf, 'w')
    json.dump(res_dict, j_f)
    print('json dump done', json_tarf)

    return json_tarf

config_file = 'configs/Mask_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_MASK_RFPN_word_margin.yaml'#'configs/ICDAR2019_det_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_RFPN.yaml' #'#"configs/ICDAR2019_det_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_4scales_angle_norm.yaml" #e2e_rrpn_R_50_C4_1x_ICDAR13_15_trial_test.yaml

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.freeze()
# cfg.MODEL.WEIGHT = 'models/IC-13-15-17-Trial/model_0155000.pth'

vis = False
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

    if merge_box:
        bboxes_np_reverse = bboxes_np.copy()
        bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
        bboxes_np_reverse = merge(bboxes_np_reverse)
        bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
        bboxes_np = bboxes_np_reverse

    width, height = bounding_boxes.size

    if vis:
        pil_image = vis_image(Image.fromarray(img), bboxes_np)
        pil_image.show()
        time.sleep(20)
    else:
        write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)
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
