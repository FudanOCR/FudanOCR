# Data  Part

- build.py: The main  processing part including getdataset and getdatadoader
- others: different types dataset from models

## dataset-type list
Here list the finished dataset
- Imdb:moran
- custom_dset: AEST
- total_text: textsnake
- icdar: maskrcnnbenchmark
- CTW1500:`psenet`
- ……
-
## Usage

You can get dataloader in following way:
```python
from data.build import build_dataloader

train_loader, test_loader = build_dataloader(env.opt)
or
test_loader = build_dataloader(env.opt, is_train = False)
```
In your config file:
you should assign DATASETS.TYPE as your datasset-type,and appoint the data_dir；

Then you can use dataloader in trainer and do more work.

## Introduce

In build.py , mainly function :
- build_dataloader: read config and return dataloader
- getdataset :  choose different dataset building strategy 
- getdataloader:  choose different loader strategy for different dataset

## ADD MORE
Add your dataset-type in getdataset and getdataloader  