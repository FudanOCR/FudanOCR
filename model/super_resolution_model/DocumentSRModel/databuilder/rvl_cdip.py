import os
import cv2
import numpy as np
import random
from tqdm import tqdm

TEMP_CACHE_NAME = './~temp.png'

gaussian_blur_params = [1, 3, 3, 3, 3, 3, 5]

def build_dataset(data_dir, new_dir='datasets', dataset_name='rvl-cdip', mode='train'):
    origin_dir = os.path.join(data_dir, dataset_name)
    label_path = os.path.join(origin_dir, 'labels', mode+'.txt')
    image_dir = os.path.join(origin_dir, 'images')
    local_dir = os.path.join(new_dir, dataset_name+'_'+mode)
    train_dir = os.path.join(new_dir, dataset_name+'_'+mode+'_train')
    valid_dir = os.path.join(new_dir, dataset_name+'_'+mode+'_valid')
    test_dir = os.path.join(new_dir, dataset_name+'_'+mode+'_test')

    if not os.path.exists(origin_dir):
        print(origin_dir)
        raise Exception('Original dataset path not exists')

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    label_file = open(label_path, 'r')
    res_dict = {}
    for idx, imgline in tqdm(enumerate(label_file)):
        res = imgline.split(' ')
        img_path, label = res[0], res[1]
        img_name = img_path.split('/')[-1]

        # load origin image
        if not os.path.exists(os.path.join(image_dir, img_path)):
            print('! Image is not exists:' + img_path)
            continue
        else:
            hr_img = cv2.imread(os.path.join(image_dir, img_path))
            
        if hr_img is None:
            print('! Image is None:' + img_path)
            continue

        if label not in res_dict.keys():
            res_dict[label] = [(img_path, img_name)]
        else: res_dict[label].append((img_path, img_name))

        # cv2.imwrite(os.path.join(local_dir, img_name), hr_img)

    idx = 0
    for key in tqdm(res_dict.keys()):
        for img_path, img_name in res_dict[key]:
            hr_img = cv2.imread(os.path.join(image_dir, img_path))
            if idx % 10 == 0:
                cv2.imwrite(os.path.join(test_dir, img_name), hr_img)
            elif idx % 10 == 1:
                cv2.imwrite(os.path.join(valid_dir, img_name), hr_img)
            else:
                cv2.imwrite(os.path.join(train_dir, img_name), hr_img)
            idx += 1

# def build_dataset(data_dir, new_dir='datasets', dataset_name='rvl-cdip', mode='train'):
#     origin_dir = os.path.join(data_dir, dataset_name)
#     label_path = os.path.join(origin_dir, 'labels', mode+'.txt')
#     image_dir = os.path.join(origin_dir, 'images')
#     local_dir = os.path.join(new_dir, dataset_name+'_'+mode)

#     if not os.path.exists(origin_dir):
#         print(origin_dir)
#         raise Exception('Original dataset path not exists')

#     if not os.path.exists(local_dir):
#         os.makedirs(local_dir)
#         os.makedirs(os.path.join(local_dir, 'LR'))
#         os.makedirs(os.path.join(local_dir, 'LRN'))
#         os.makedirs(os.path.join(local_dir, 'HR'))

#     label_file = open(label_path, 'r')
#     for idx, imgline in tqdm(enumerate(label_file)):
#         img_path = imgline.split(' ')[0]
#         img_name = img_path.split('/')[-1]

#         # load origin image
#         if not os.path.exists(os.path.join(image_dir, img_path)):
#             print('! Image is not exists:' + img_path)
#             continue
#         else:
#             hr_img = cv2.imread(os.path.join(image_dir, img_path))
            
#         if hr_img is None:
#             print('! Image is None:' + img_path)
#             continue

#         # build general low resolution image
#         lr_img = cv2.resize(hr_img, None, None, 0.5, 0.5)
#         lrn_img = lr_img.copy()

#         # build noisy low resolution image
#         prob = random.random()
#         if prob <= 0.45:
#             degradation = 'compression'
#         elif prob <= 0.85:
#             degradation = 'gaussian blur'
#         elif prob <= 0.7:
#             degradation = 'gaussian noise'
#         elif prob < 0.8:
#             degradation = 'salt pepper noise'
        
#         # additional degradation 
#         if degradation == 'compression':
#             r1 = np.random.randint(5, 95)
#             r2 = np.random.randint(2, 10)
#             cv2.imwrite(TEMP_CACHE_NAME, lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), r1])
#             lrn_img = cv2.imread(TEMP_CACHE_NAME)
#             cv2.imwrite(TEMP_CACHE_NAME, lrn_img, [int(cv2.IMWRITE_PNG_COMPRESSION), r2])
#             lrn_img = cv2.imread(TEMP_CACHE_NAME)

#         elif degradation == 'gaussian blur':
#             r = int(np.random.choice(gaussian_blur_params))
#             lrn_img = cv2.GaussianBlur(lr_img, (r, r), 0)

#         elif degradation == 'salt pepper noise':
#             pass

#         cv2.imwrite(os.path.join(local_dir, 'HR', img_name), hr_img)
#         cv2.imwrite(os.path.join(local_dir, 'LR', img_name), lr_img)
#         cv2.imwrite(os.path.join(local_dir, 'LRN', img_name), lrn_img)

#     if os.path.exists(TEMP_CACHE_NAME):
#         os.remove(TEMP_CACHE_NAME)