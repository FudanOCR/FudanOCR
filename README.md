# 复旦OCR系统

------

此系统集成若干OCR方法，包括检测、识别、端到端框架，旨在为研究人员提供便利性。系统包括的模型既有2019年ICDAR比赛中使用的模型，还有17级毕业师兄师姐毕业论文使用的模型。模型跑通后的实验数据记录在共享文档中，链接如下

### [记录实验数据的共享文档](https://docs.qq.com/desktop/mydoc/folder/aE338MoFVm_100001)

### 注意事项 :wink:
* /train下的文件最好不要在函数外包含import语句，否则会出现例如执行MORAN_V2模型却要安装GRCNN模型的相关包
* 在train文件夹下编写文件时，文件头请加入# -*- coding: utf-8 -*-


### 主要文件
> * /config   配置文件，里面应该包含model参数指定使用的模型
> * /technical_report 技术报告，包括复旦OCR白皮书与若干毕业论文
> * /documents 各种模型配置文档
> * /detection_model 检测模型
> * /recognition_model 识别模型
> * /end2end_model 结合检测和识别功能的端到端模型
> * /maskrcnn_benchmark_architecture 使用开源架构的模型
> * /train  主方法从该文件夹中导入训练模型的方法
> * /val 主方法从该文件夹中导入测试模型的方法
> * /demo 实验结果可视化
> * main.py 使用python main.py --config_file xxx 传入配置文件训练模型  

### 导入模型需要修改main.py的部分   

```python
import re
import argparse
from yacs.config import CfgNode as CN

from train.moran_v2 import train_moran_v2
# 添加 from train.文件名 import 模型训练函数

# 在这个位置扩充函数
function_dict = {

    'MORAN': train_moran_v2,
    # 添加'Your Model Name': 'Your Model Function'
}
```


