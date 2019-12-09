# 主函数入口
# 这个文件的目的是调用需要的模型，进行训练，测试功能
# 后期需要一直扩展下去
# 加油~


from moran_v2 import train_moran_v2



if __name__ == '__main__':
    # coding:utf8
    import os

    # 获取当前目录绝对路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    print('当前目录绝对路径:', dir_path)