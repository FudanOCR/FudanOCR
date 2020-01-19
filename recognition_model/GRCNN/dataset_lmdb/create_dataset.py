#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
create_dataset.py文件用于生成数据集的lmdb文件，以供dataset.py中的lmdbDataset使用
'''


import os
import lmdb  # install lmdb by "pip install lmdb"
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
        print("检查")
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)



def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    # imagePathList 瀵瑰簲鍜?labelList 閰嶅
    assert (len(imagePathList) == len(labelList))

    # 鑾峰彇鏁伴噺
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        # 鍒ゆ柇鏄惁瀛樺湪璺緞
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        # 鐩存帴璇诲彇鍥剧墖
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
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_image_label(image_directory, label_address):
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
    i = 1

    # 图片：目标记录

    for line in f.readlines():
        # TODO
        dict[line[10:].split(" ")[0]] = line.split(' ')[1].replace('\n', '').replace('\r',
                                                                                     '')  # arttrain-11.art/lsvttest10.lsvt12
        '''
        print(dict)

        i+=1
        if i==14:
            break
        print(dict)
        '''

    # print(dict)
    result1 = []
    result2 = []
    # TODO

    for image_path1 in image_lis:
        for image_path2 in os.listdir(image_directory + '/' + image_path1):

            try:
                # image_path = image_path.replace('.jpg','')
                # result1.append(image_directory+'/'+image_path1+'/'+image_path2)
                result2.append(dict[image_path1 + '/' + image_path2])
                result1.append(image_directory + '/' + image_path1 + '/' + image_path2)
            except:
                # pass
                print("jianzhi")

    return result1, result2


def extract_result_from_xml():
    import re

    f = open('../xml_test/word.xml', 'r')

    string = ""
    for line in f.readlines():
        print(line)
        string += line

    print(string)

    # 记录文件路径
    result1 = re.findall(r'file=\"(.*?)\"', string)

    for i in range(len(result1)):
        result1[i] = '/home/chenjingye/datasets/ICDAR2003/WordR/TrialTest/' + result1[i]


    print(result1)

    result2 = re.findall(r'tag=\"(.*?)\"', string)
    print(result2)

    return result1, result2

def ic15():
    f = open('/home/chenjingye/datasets/ICDAR2015/Word_recognition/Challenge4_Test_Task3_GT.txt', 'r')

    result1 = []
    result2 = []
    for line in f.readlines():
        # print(line)
        # print(line.split())
        a, b = line.split(', ')
        print(a, b)
        result1.append('/home/chenjingye/datasets/ICDAR2015/Word_recognition/ch4_test_word_images_gt/' + a.replace(',', ''))
        result2.append(b.replace("\"", "").replace('\r\n',''))

    print(result1)
    print(result2)
    return result1 , result2


def find_jpg():

    import os

    # dataset_add = "/mnt/sdb1/zifuzu/chenjingye/datasets/mnt/ramdisk/max/90kDICT32px"
    # files = os.listdir(dataset_add)
    #
    # cnt = 0
    # for file in files:
    #     jpg_file = os.listdir(os.path.join(dataset_add,file))
    #
    #
    #     for next_jpg_file in jpg_file:
    #         # os.listdir(os.path.join(dataset_add, file,
    #         print(next_jpg_file)
    #
    # print(files)

    root = "/mnt/sdb1/zifuzu/chenjingye/datasets/mnt/ramdisk/max/90kDICT32px"

    flag = True

    def findtxt(path, ret):
        """Finding the *.txt file in specify path"""
        filelist = os.listdir(path)
        for filename in filelist:

            if len(ret) > 100000 :
                return

            de_path = os.path.join(path, filename)
            if os.path.isfile(de_path):
                if de_path.endswith(".jpg"):  # Specify to find the txt file.
                    print(de_path)
                    ret.append(de_path)
                if len(ret) > 100000:
                    return
            else:
                findtxt(de_path, ret)

    ret = []
    findtxt(root, ret)
    for path in ret:
        print(path)

    # 删除一个temp.txt，再创建一个txt
    try:
        os.remove('./temp.txt')
    except:
        pass
    f = open('./temp.txt','a')
    for element in ret:
        f.write(element + '\n')
    f.close()


def syn90():

    import re
    f = open('./temp.txt','r')

    result1 = []
    result2 = []

    for line in f.readlines():
        result1.append(line.replace('\n',''))

        target = re.findall(r'_(.*?)_',line)[0]
        result2.append(target)

    return result1, result2



if __name__ == '__main__':
    find_jpg()
    # TODO
    #result1, result2 = read_image_label('./ArTtest', './ArTtest.txt')
    result1, result2 = syn90()
    print(result1)
    print(result2)
    createDataset('/mnt/sdb1/zifuzu/chenjingye/datasets/syn90_train_100000data_lmdb', result1, result2)



