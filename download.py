import prettytable as pt
import argparse
import urllib.request


class Resource():
    def __init__(self, name, path, size, description):
        self.name = name
        self.path = path
        self.size = size
        self.description = description

    def download(self, des_root):
        if des_root == '.':
            des_root = './' + self.path.split('/')[-1]
        urllib.request.urlretrieve(self.path, des_root, self.callbackfunc)
        print('\n Finish! \n')

    def callbackfunc(self, blocknum, blocksize, totalsize):
        '''回调函数
        @blocknum: 已经下载的数据块
        @blocksize: 数据块的大小
        @totalsize: 远程文件的大小
        '''
        percent = 100.0 * blocknum * blocksize / totalsize

        nowLoad = blocknum * blocksize / 1024 / 1024
        total = totalsize / 1024 / 1024

        if percent > 100:
            percent = 100

        print("\r %.2fM/%.2fM    %.2f%%" % (nowLoad, total, percent), end=" ")


resourceDict = {

    'ICDAR2003': Resource(
        name='ICDAR2003',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ICDAR2003.zip',
        size='413M',
        description='R/D Dataset'
    ),
    'ICDAR2013': Resource(
        name='ICDAR2013',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ICDAR2013.zip',
        size='159M',
        description='R/D Dataset'
    ),
    'ICDAR2015': Resource(
        name='ICDAR2015',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ICDAR2015.zip',
        size='250M',
        description='R/D Dataset'
    ),
    'Syn800w': Resource(
        name='Syn800w',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_9000000.tar.gz',
        size='9.9G',
        description='R Dataset'
    ),
    'TotalText': Resource(
        name='TotalText',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_totaltext_detection.zip',
        size='412M',
        description='D Dataset'
    ),
    'LSVT_TRAIN': Resource(
        name='LSVT_TRAIN',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_LVST_TRAIN.zip',
        size='1.0G',
        description='R Dataset'
    ),
    'LSVT_TEST': Resource(
        name='LSVT_TEST',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_LVST_TEST.zip',
        size='115.3M',
        description='R Dataset'
    ),
    'LSVT_LABEL': Resource(
        name='LSVT_LABEL',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_LVST_LABEL.zip',
        size='2.6M',
        description='R Dataset'
    ),
    'ART_TRAIN': Resource(
        name='ART_TRAIN',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ART_TRAIN.zip',
        size='285.6M',
        description='R Dataset'
    ),
    'ART_TEST': Resource(
        name='ART_TEST',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ART_TEST.zip',
        size='33M',
        description='R Dataset'
    ),
    'ART_LABEL': Resource(
        name='ART_LABEL',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/DATASET_ART_LABEL.zip',
        size='451K',
        description='R Dataset'
    ),
    'tomcat': Resource(
        name='tomcat',
        path='https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/apache-tomcat-9.0.33.zip',
        size='15.1M',
        description='Web Server'
    ),

}

parser = argparse.ArgumentParser()
# parser.add_argument('--all', action='store_true', help='List all available resources')
parser.add_argument('--name', type=str, help='Resource name', default='null')
parser.add_argument('--path', type=str, help='The path where you want to download the resource to', default='.')
opt = parser.parse_args()
# print(opt)

if True:
    '''用户输入指令 python download.py --all '''

    tb = pt.PrettyTable()
    tb.field_names = ["Name", "Path", "Size", "Description"]
    for key in resourceDict.keys():
        resource = resourceDict[key]
        tb.add_row([resource.name, resource.path, resource.size, resource.description])

    print(tb)

if opt.name != 'null':
    if opt.name not in resourceDict.keys():
        assert False, 'Resource not found!'
    else:
        resourceDict[opt.name].download(opt.path)
