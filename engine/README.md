# Engine  Module

- env.py: Initialize your training environment and read the config file from the command line.
- loss.py: Get loss functions for your trainer. Add your specialize loss function in /personalize_loss.
- optimizer.py: Get optimizers for your trainer. Add your specialize optimizer in /personalize_optimizer.
- pretrain.py: Define a dictionary for (model_name,pretrain_address), your can decide whether to use pretrained model in config file.
- trainer.py: Define a class Trainer. Use that class in main.py and you need to overload some of the funcitons to achieve yours target.

## Usage

To initialize a new training environment, just creat a new Env object!
```python
from engine.env import Env
env = Env()
opt = env.getOpt() 
'''Use opt to get parameters from config file
print(opt.BASE.MODEL)
'''
```

Your can define a sub-class of Trainer in main.py
```python
from engine.trainer import Trainer
class XXNET_Trainer(Trainer):
    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data):
        '''You need to overload'''
        pass

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''You need to overload'''
        pass
```

After declaring dataset and model, use Trainer.train() to run.
```python
train, test  = build_dataloader(env.opt)
newTrainer = XX_Trainer(modelObject=model, opt=env.opt, train_loader=train, val_loader=test).train()
```

