import os
import cv2
import numpy as np

if __name__ == '__main__':
    path = '/home/shf/fudan_ocr_system/textsnake_pytorch/output/ICDAR19/'

    train_files = os.listdir(os.path.join(path, 'train_images'))
    test_files = os.listdir(os.path.join(path, 'test_images'))

    mean_r = []
    mean_g = []
    mean_b = []
    std_r = []
    std_g = []
    std_b = []

    for i, file in enumerate(train_files):
        print(i)
        img = cv2.imread(os.path.join(path, 'train_images', file))
        img = img / 255.0
        mean_b.append(np.mean(img[:, :, 0]))
        mean_g.append(np.mean(img[:, :, 1]))
        mean_r.append(np.mean(img[:, :, 2]))
        std_b.append(np.std(img[:, :, 0]))
        std_g.append(np.std(img[:, :, 1]))
        std_r.append(np.std(img[:, :, 2]))

    global_mean_b = np.mean(mean_b)
    global_mean_g = np.mean(mean_g)
    global_mean_r = np.mean(mean_r)
    global_std_b = np.mean(std_b)
    global_std_g = np.mean(std_g)
    global_std_r = np.mean(std_r)

    print(global_mean_r, global_mean_g, global_mean_b)
    print(global_std_r, global_std_g, global_std_b)

