import os
import cv2
import numpy as np

if __name__ == '__main__':
    data_root = os.path.join('/home/shf/fudan_ocr_system/datasets/ICDAR19', 'train_images')
    files = os.listdir(data_root)

    global_mean_r = []
    global_mean_g = []
    global_mean_b = []
    global_std_r = []
    global_std_g = []
    global_std_b = []

    for i, file in enumerate(files):
        print(i+1)
        img = cv2.imread(os.path.join(data_root, file))
        img = img / 255.

        global_mean_b.append(np.mean(img[:, :, 0]))
        global_mean_g.append(np.mean(img[:, :, 1]))
        global_mean_r.append(np.mean(img[:, :, 2]))
        global_std_b.append(np.std(img[:, :, 0]))
        global_std_g.append(np.std(img[:, :, 1]))
        global_std_r.append(np.std(img[:, :, 2]))

    mean_b = np.mean(global_mean_b)
    mean_g = np.mean(global_mean_g)
    mean_r = np.mean(global_mean_r)
    std_b = np.mean(global_std_b)
    std_g = np.mean(global_std_g)
    std_r = np.mean(global_std_r)

    print(mean_r, mean_g, mean_b)
    print(std_r, std_g, std_b)
