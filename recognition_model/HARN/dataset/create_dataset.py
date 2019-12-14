#!/usr/bin/python
# coding=UTF-8

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tools import dataset as dataset
import torch


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
    	imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    	img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    	imgH, imgW = img.shape[0], img.shape[1]
    	if imgH * imgW == 0:
        	return False
    except:
        print("checking is wrong")
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

       
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
       
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin

        cache[labelKey] = label


        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_image_label(image_directory,label_address):
    import os
    image_lis = os.listdir(image_directory)  # image_lis是图像文件的名字列表

    f = open(label_address)
    dict = {}
    for line in f.readlines():
        # TODO
        # print("1    " + line[54:])
        # print(line[54:].split(" "))  # 将截取出的字符串按空格分割
        # print("3    " + line[54:].split(" ")[0])   # 取出其中的第一个，应该是想要 1.jpg 这种吧
        # print(line.split(' ')[1].replace('\n','').replace('\r','')) # 取出第二个数据，将换行符换为空格，windows的换行符是"\r\n"
        dict[line[23:].split(" ")[0]] = line.split(' ')[1].replace('\n','') # 前54个字符都不要 #######可能需要改
        # dict[line[54:].split(" ")[0]] = line.split(' ')[1].replace('\n','').replace('\r','')  # 前54个字符都不要 #######可能需要改
    #print(dict)
    result1 = []
    result2 = []
    # TODO
    for image_path1 in image_lis:
        # print(image_directory+'/'+image_path1)   # 每张图片中可能有多个文字区域，所以图像名称目录存放检测结果
        for image_path2 in os.listdir(image_directory+'/'+image_path1):

            try:
            # image_path = image_path.replace('.jpg','')
                # result1.append(image_directory+'/'+image_path1+'/'+image_path2)
                result2.append(dict[image_path1+'/'+image_path2])
                result1.append(image_directory+'/'+image_path1+'/'+image_path2)
            except:
                # pass
                print("key is not match")

    return result1,result2


def read_lmdb():
    train_nips_dataset = dataset.lmdbDataset(root='/mnt/sdb1/zifuzu/miaosiyu/datasets/OCR_dataset/test')
    # train_nips_dataset = dataset.lmdbDataset(root='./lsvt_test')  # 转换test的路径    # 哪里引用了???
    return train_nips_dataset


if __name__ == '__main__':
    # TODO
    result1,result2 = read_image_label('/home/miaosiyu/code_dataset/dataset/OCR_dataset/pre_1000','/home/miaosiyu/code_dataset/dataset/OCR_dataset/val.txt')  # (image_directory, label_address)
    # result1,result2 = read_image_label('/home/miaosiyu/code_dataset/dataset/OCR_dataset/pre_9000','/home/miaosiyu/code_dataset/dataset/OCR_dataset/train.txt')  # (image_directory, label_address)
    # result1,result2 = read_image_label('/home/cqy/art/ArTtrain','./ArTtrain.txt')
    createDataset('/mnt/sdb1/zifuzu/miaosiyu/datasets/OCR_dataset/test',result1,result2)





