# 创建数据集
- 使用python2环境执行 ./dataset/create_dataset.py 
- read_image_label函数：返回两个list：result1，result2，分别为所有图片的路径、所有图片的标签，并且在两个列表的相同位置是一一对应的
- createDataset函数：将read_image_label函数返回的两个list转变为lmdb数据库格式
- 建议先阅读create_dataset.py中的代码，特别是3个TODO，合理改变存放的路径
