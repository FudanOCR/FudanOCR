# -*- coding:utf-8 -*-
import time
import numpy as np
import visdom
import torchvision

class Visualizer():
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.names = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d, win):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        name=list(d.keys())
        val=list(d.values())
        x = self.index.get(win, 0)
        
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(win),
                    opts=dict(legend=name,
                        title=win),
                    update=None if x == 0 else 'append'
                    )
        self.index[win] = x + 1

    def img_many(self, d):
        '''
        draw several images
        '''
        for k, v in d.items():
            self.img(k, v)

    def plot(self, win, y, name=None):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(win, 0)
        opts = dict(title=win)
        if name is not None:
            opts['legend'] = name
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=win,
                      opts=opts,
                      update=None if x == 0 else 'append')
        self.index[win] = x + 1    

    def img(self, name, img_):
        '''
        self.img('input_img',t.Tensor(64,64))
        '''
        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name))


    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        '''
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        '''
        self.img(name, torchvision.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1,min=0)))

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),\
            info=info))
        self.vis.text(self.log_text, win=win)

    def metrics(self, metrics, win):
        out = ""
        for key, val in metrics.items():
            out += "{}: {} <br>".format(key, val)
        self.vis.text(out, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)