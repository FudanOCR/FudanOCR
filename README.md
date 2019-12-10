# 复旦OCR系统

------

此系统集成若干OCR方法，包括检测、识别、端到端框架，旨在为研究人员提供便利性。系统包括的模型既有2019年ICDAR比赛中使用的模型，还有17级毕业师兄师姐毕业论文使用的模型。模型跑通后的实验数据记录在共享文档中，链接如下

### [记录实验数据的共享文档](https://docs.qq.com/desktop/mydoc/folder/aE338MoFVm_100001)



### 主要文件
> * /config   配置文件，里面应该包含model参数指定使用的模型
> * /train  主方法从该文件夹中导入训练模型的方法
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
