#coding:utf-8
from PIL import Image
import ssim
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math

from scipy.signal import convolve2d


def PSNR(im1, im2):
    mse = np.mean( np.abs((im1/255. - im2/255.)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX/ np.sqrt(mse))

def SSIM(im1, im2):
    im1 = Image.fromarray(im1.transpose(1, 2, 0))
    im2 = Image.fromarray(im2.transpose(1, 2, 0))
    return ssim.SSIM(im1).cw_ssim_value(im2)
    # ssim_val = 0
    # for i in range(im1.shape[0]):
    #     ssim_val += compute_ssim(im1[i], im2[i])
    # return ssim_val / im1.shape[0]


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def batch_SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding = int(window_size/2), groups = channel)
    mu2 = F.conv2d(img2, window, padding = int(window_size/2), groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = int(window_size/2), groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = int(window_size/2), groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = int(window_size/2), groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def batch_compare_filter(batch1, batch2, func):
    def tensor_to_np(tensor):
        dim = tensor.dim()
        tensor = tensor.cpu().clone().mul(255).byte()
        if dim == 4:
            img = tensor.numpy()
            res = img
        else:
            img = tensor.unsqueeze(0).numpy()
            res = img
        return res.astype(np.uint8)
    # print(batch1.size(), batch2.size())
    batch1 = tensor_to_np(batch1)
    batch2 = tensor_to_np(batch2)
    score = 0.0
    size = list(batch1.shape)
    for i in range(size[0]):
        score += func(batch1[i], batch2[i])
    score /= size[0]
    return score