version = "2s"
epoch = 60000
learning_rate = 1e-3
learning_rate2 = 1e-2
all_trains = 1000
batch_size = 12
momentum = 0.9
weight_decay = 5e-4
dilation = True
use_crop = False
use_rotate = True
# iterations = 10
gpu = True
multi_gpu = True  # only useful when gpu=True
pixel_weight = 2
link_weight = 1

r_mean = 123.
g_mean = 117.
b_mean = 104.

image_height = 512
image_width = 512
image_channel = 3

link_weight = 1
pixel_weight = 2
neg_pos_ratio = 3  # parameter r in paper

train_images_dir = "/home/msy/datasets/ICDAR15/Text_Localization/train/img/"
train_labels_dir = "/home/msy/datasets/ICDAR15/Text_Localization/train/gt/"
saving_model_dir = "/home/msy/PixelLink-with-pytorch/model/"
retrain_model_index = 26200  # retrain from which model, e.g. ${saving_model_dir}/156600.mdl
test_model_index = 43800  # test for which model, e.g. ${saving_model_dir}/156600.mdl
test_batch = 1

retrain_epoch = 60000
retrain_learning_rate = 1e-2
retrain_learning_rate2 = 3e-3

