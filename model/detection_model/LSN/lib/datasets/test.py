import cv2
import os
from imgaug import augmenters as iaa
import numpy as np
import random

#===================================================#
# test imguag
# path = '/data/2019AAAI/data/ctw15/test/text_image'
# # blurer = iaa.Invert(0.25, per_channel=0.5)
# # blurer = iaa.Multiply((0.5, 1.5))
# while True:
#     for imagename in os.listdir(path):
#         image = cv2.imread(os.path.join(path,imagename))
#         seg = iaa.Sequential([iaa.Invert(random.randint(0,100)*1.0/100),iaa.EdgeDetect(random.randint(0,10)*1.0/10)],random_order=True)
#         iaaimage = seg.augment_images([image])[0]
#         cv2.imshow('image',image)
#         cv2.imshow('iaaimage',iaaimage)
#         cv2.waitKey(0)


#===================================================#
# test rectangle overlap

def is_rect_overlap(r1,r2):
    return not (((r1[2] < r2[0]) | (r1[3] > r2[1])) | ((r2[2] < r1[0]) | (r2[3] > r1[1])))

if __name__ == '__main__':
    rec1 = np.array([[50,50,100,100]])
    rec2 = np.array([[0,0,150,150],[1000,1000,1200,1200],[1000,1000,1200,1200],[0,0,150,150]])
    print(is_rect_overlap(rec1,rec2))