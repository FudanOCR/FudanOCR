from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
from torchvision import transforms
import torch
import cv2
import numpy as np
from torch.autograd import Variable

def to_varabile(data,requires_grad,is_cuda):
    if is_cuda:
        data = data.cuda()
    data = Variable(data,requires_grad=requires_grad)
    return data

# input data
is_cuda = torch.cuda.is_available()
# image_data = cv2.imread('/data/2019AAAI/data/ctw15/test/text_image/1002.jpg')
image_data = np.ones((100,100,3))
image_data = image_data.transpose((2, 0, 1)).astype(np.float32)
image_data = torch.from_numpy((image_data))
boxes_data = torch.Tensor([[0,0,200,200],[0,0,200,200]])
box_index_data = torch.IntTensor([0])
image = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
image = image.unsqueeze(0)
print(image.size())
boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)
print(image,boxes,box_index)
# RoIAlign layer
roi_align = RoIAlign(7, 7,extrapolation_value=0)
crops = roi_align(image, boxes, box_index)
print(crops)

