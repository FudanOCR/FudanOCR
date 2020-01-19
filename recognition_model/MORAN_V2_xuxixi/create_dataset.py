#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
create_dataset.py文件用于生成数据集的lmdb文件，以供dataset.py中的lmdbDataset使用
'''

import os
import lmdb
import cv2
import numpy as np
import tools.dataset as dataset
import torch


def checkImageIsValid(imageBin):
    '''
    核对图片的二进制格式是否合理
    通过将string还原为图片格式，如果图片的长宽相乘为0，则该图片错误，返回False
    '''
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
    except:
        print("检查出错")
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

# outputPath:输出路径
# imagePathList:图片路径List
# labelList:标签List
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    # imagePathList 对应和 labelList 配套
    assert(len(imagePathList) == len(labelList))

    # 获取数量
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        # 判断是否存在路径
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        # 直接读取图片
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
    '''
    :param str image_directory 图片的目录
    :param str label_address 标签的目录
    :return list result1 每张图片目录的列表
    :return list result2 每张图片标签的列表，其中result1和result2在相同索引的图片-标签是成对的
    '''
    import os
    image_lis = os.listdir(image_directory)

    f = open(label_address)
    dict = {}
    i=1
    for line in f.readlines():
        # TODO
        dict[line[11:].split(" ")[0]] = line.split(' ')[1].replace('\n','').replace('\r','')


    #print(dict)
    result1 = []
    result2 = []
    # TODO
    
    for image_path1 in image_lis:
        for image_path2 in os.listdir(image_directory+'/'+image_path1):

            try:
            # image_path = image_path.replace('.jpg','')
                # result1.append(image_directory+'/'+image_path1+'/'+image_path2)
                result2.append(dict[image_path1+'/'+image_path2])
                result1.append(image_directory+'/'+image_path1+'/'+image_path2)
            except:
                # pass
                print("键值对未匹配")
    
    return result1,result2
    
'''
def read_lmdb():
    # 读取IMDB格式的数据，并调整大小
    train_nips_dataset = dataset.lmdbDataset(root='./temp_output')
    return train_nips_dataset
'''

if __name__ == '__main__':
    # TODO
    result1,result2 = read_image_label('./ArTtrain','./ArTtrain.txt')
    createDataset('./art_train',result1,result2)





