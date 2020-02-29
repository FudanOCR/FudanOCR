# -*- coding: utf-8 -*-

def _init():#初始化
    global _det_val_result
    _det_val_result = {}

def _reset():
    _det_val_result.clear()

def _update(dict_a):
    """ 定义一个全局变量 """
    _det_val_result.update(dict_a)

def _get_det_value():
    """" 获得一个全局变量,不存在则返回默认值 """
    return _det_val_result