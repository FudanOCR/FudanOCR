# A PyTorch re-implementation of AdvancedEAST

[GitHub](https://github.com/huoyijie/AdvancedEAST/issues)

## Environment

* python 3.7
* pytorch 1.0.1
* numpy 1.16.2
* cython 0.29.5
* pillow 5.4.1
* tqdm 4.31.1

## 原理简介

**网络输出**

输出层分别是1位score map, 是否在文本框内；2位vertex code，是否属于文本框边界像素以及是头还是尾；4位geo，是边界像素可以预测的2个顶点坐标。所有像素构成了文本框形状，然后只用边界像素去预测回归顶点坐标。边界像素定义为黄色和绿色框内部所有像素，是用所有的边界像素预测值的加权平均来预测头或尾的短边两端的两个顶点。头和尾部分边界像素分别预测2个顶点，最后得到4个顶点坐标。

[原理简介](https://huoyijie.github.io/zh-Hans/2018/08/24/AdvancedEAST%E6%96%87%E6%9C%AC%E6%A3%80%E6%B5%8B%E5%8E%9F%E7%90%86%E7%AE%80%E4%BB%8B/)

[后置处理](https://huoyijie.github.io/zh-Hans/2018/08/27/AdvancedEAST%E5%90%8E%E7%BD%AE%E5%A4%84%E7%90%86%E5%8E%9F%E7%90%86%E7%AE%80%E4%BB%8B/)
