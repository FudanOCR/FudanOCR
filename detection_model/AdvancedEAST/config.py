import os
# === Task ===

task_id = '3T1280'

train_size = int(task_id[2:])
size_group = [512, 640, 768, 896, 1024, 1280, 1408, 1536, 1920]
assert train_size in size_group, f'input size shall be in {size_group}'

# === Dataset ===
dset_name = 'ICADR15'
train_img = '/home/msy/ICDAR15/Text_Localization/train/img/'
train_gt = '/home/msy/ICDAR15/Text_Localization/train/gt/'
val_img = '/home/msy/ICDAR15/Text_Localization/val/img/'
val_gt = '/home/msy/ICDAR15/Text_Localization/val/gt/'

# dset_name = 'LSVT'
# train_img = './dataset/LSVT/train/img/'
# train_gt = './dataset/LSVT/train/gt/'
# val_img = './dataset/LSVT/val/img/'
# val_gt = './dataset/LSVT/val/gt/'
num_process = max(os.cpu_count() - 2, os.cpu_count() // 2)  # for DataLoader
cache_dir = './cache/'
result_dir = './result/'

# === Training ===

pre_file = './result/3T1280_archive/3T1280_center.pth.tar'  # file path

# batch_size = 1
# batch_size_group = [1, 1, 1, 1, 1, 1, 1, 1, 1]
batch_size_group = [15, 10, 6, 4, 4, 2, 1, 1, 1]
batch_size_per_gpu = batch_size_group[size_group.index(train_size)]

# gpu_ids = [1, 2]
# gpu_ids = [0, 1, 2, 3]
gpu_ids = [0, 1]

lr_rate = 0.0001
decay_step = 45  # decay every x epoch
decay_rate = 0.1

max_epoch = 50
patience = 5
print_step = 270  # print every x iter

num_workers = 4
init_type = 'xavier'

# === Parameters ===

lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

shrink_ratio = 0.2  # in paper it's 0.3, maybe too large in the case
shrink_side_ratio = 0.6  # pixels between shrink_ratio and shrink_side_ratio are side pixels
epsilon = 1e-4

pixel_size = 4

# === Evaluation ===
gt_json_path = './dataset/LSVT/train_full_labels.json'   # 没有，自己生成
iou_threshold = 0.5

pixel_threshold = 0.9  # pixel activation threshold
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
