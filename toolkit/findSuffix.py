# -*- coding: utf-8 -*-
import os

# 给定根路径与后缀名，寻找根路径下所有以该后缀名结尾的文件，并返回至return数组
# suffix: 后缀名，例如 'txt'
# maxlen: 返回数组的最大长度
# path: 根路径
# ret: 返回数组
def findSuffix(suffix, maxlen , path, ret):
    """Finding the *.suffix file in specify path"""
    filelist = os.listdir(path)
    for filename in filelist:

        if len(ret) > maxlen :
            return

        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(suffix):  # Specify to find the txt file.
                print(de_path)
                ret.append(de_path)
            if len(ret) > maxlen:
                return
        else:
            findtxt(suffix, maxlen, de_path, ret)

