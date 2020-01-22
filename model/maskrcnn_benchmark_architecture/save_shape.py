import os
import cv2

img_dir = '../datasets/LSVT/train_full_images_0/train_full_images_0/'
output_file = './shape_log.txt'

shape_str = ''

for i in range(3000):
    im_name = 'gt_' + str(i) + '.jpg'
    im_path = os.path.join(img_dir, im_name)
    print('impath:', im_path)
    im = cv2.imread(im_path)
    shape_str += im_name + ' ' + str(im.shape[0]) + ' ' + str(im.shape[1]) + '\n'

print('Writing log...')
log_f = open(output_file, 'w')
log_f.write(shape_str)
log_f.close()
print('done')