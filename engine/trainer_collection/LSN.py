from engine.trainer import Trainer
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import cv2
import math
import os


class LSN_Trainer(Trainer):

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def get_img_data(self, pretreatmentData):
        tensor, all_circle,res, sample = pretreatmentData
        return (tensor, all_circle, None)

    def rescale(self, image, preferredShort=768, maxLong=2048):
        h, w, _ = image.shape
        longSide = max(h, w)
        shortSide = min(h, w)
        self.ratio = preferredShort * 1.0 / shortSide
        if self.ratio * longSide > maxLong:
            self.ratio = maxLong * 1.0 / longSide
        image = cv2.resize(image, None, None, self.ratio, self.ratio, interpolation=cv2.INTER_LINEAR)
        return image

    def alignDim(self, image):
        h2, w2, _ = image.shape
        H2 = int(math.ceil(h2 / 64.0) * 64)
        W2 = int(math.ceil(w2 / 64.0) * 64)
        ret_image = np.zeros((H2, W2, _))
        ret_image[:h2, :w2, :] = image
        return ret_image

    def pretreatment(self, data, test=False):
        if test == False:

            image = data['image']


            circle_labels = {}
            image = Variable(image.cuda(), requires_grad=False)
            all_circle = {}
            all_mask = {}
            mask_gt = {}
            for stride in self.opt.LSN.strides:
                circle_labels[str(stride)] = Variable(data['labels_stride_' + str(stride)].squeeze().cuda(),
                                                      requires_grad=False)
                all_circle[str(stride)] = Variable(data['anchors_stride_' + str(stride)].squeeze().cuda(),
                                                   requires_grad=False)
                all_mask[str(stride)] = Variable(data['mask_stride_' + str(stride)].squeeze().cuda(), requires_grad=False)

            return (image, all_circle, all_mask, circle_labels, mask_gt)
        else:
            image = data['image']
            image = self.rescale(image)
            image = self.alignDim(image)
            sample = {}
            ptss = [
                [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                 [1, 1], [1, 1], [1, 1]]]
            bbx = [[1, 1, 1, 1]]
            all_circle = {}
            for stride in self.strides:
                labels = None
                all_anchors = None
                labels, all_anchors, mask_label = self.PG.run(stride, np.array([2, 2.5, 3, 3.5]), [1],
                                                              image.shape[0] / stride, image.shape[1] / stride,
                                                              [image.shape[0], image.shape[1], 1], 0, ptss, image, bbx)
                sample[str(stride)] = all_anchors
                all_circle[str(stride)] = Variable(
                    torch.from_numpy(np.ascontiguousarray(sample[str(stride)].astype(np.float32))).squeeze().cuda(),
                    requires_grad=False)
            tensorimage = image.copy()
            tensor = self.toTensor(tensorimage).unsqueeze(0)
            tensor = Variable(tensor.cuda(), requires_grad=False)
            res = None
            self.model.eval()
            threshold = 0.5
            testMaxNum = 1000

            return (tensor, all_circle,res, sample)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        if test == False:
            circle_labels_pred, pred_mask, bbox, bbox_idx, pos_idx_stride, bbox_score = modelResult
            image, all_circle, all_mask, circle_labels, mask_gt = pretreatmentData
            loss = None

            losses = {}
            mask_label = None
            for stride in self.strides:
                # circle cls
                pred_labels = circle_labels_pred[str(stride)]
                target_labels = circle_labels[str(stride)]
                label_temploss = None
                if str(stride) in pos_idx_stride:
                    stride_mask = all_mask[str(stride)][pos_idx_stride[str(stride)]]
                    if type(mask_label) == type(None):
                        mask_label = stride_mask
                    else:
                        mask_label = torch.cat((mask_label, stride_mask), 0)
                label_temploss = self.anchor_cls_loss_function(pred_labels, target_labels)
                # if self.net == 'resnet50_mask':
                #     losses['seg_' + str(stride)] = F.smooth_l1_loss(mask_labels[str(stride)], mask_gt[str(stride)])
                #     Epoch_image_mask_loss[stride] = Epoch_circle_cls_Loss[stride] + toNp(losses['seg_' + str(stride)])
                losses['cls_' + str(stride)] = label_temploss
                # Epoch_circle_cls_Loss[stride] = Epoch_circle_cls_Loss[stride] + toNp(losses['cls_' + str(stride)])

            if not type(mask_label) == type(None):
                mask_label = mask_label.squeeze()
                ## show mask
                pred_mask = pred_mask
                losses['mask'] = F.smooth_l1_loss(pred_mask, mask_label)
                # Epoch_mask_loss = Epoch_mask_loss + toNp(losses['mask'])
            for key in losses:
                if type(loss) == type(None):
                    loss = losses[key]
                else:
                    loss += losses[key]

            # loss = losses['mask']
            # print(loss)
            # print(Epoch_circle_cls_Loss,Epoch_circle_reg_Loss)
            return loss
        else:
            circle_labels_pred, pred_mask, bbox, bbox_idx, pos_idx_stride, bbox_score = modelResult
            image, all_circle, all_mask, circle_labels, mask_gt = pretreatmentData
            pred_mask_all = pred_mask.data.cpu().numpy()
            image = image.astype(np.uint8)
            showimage = image.copy()
            showmask = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
            score_list = bbox_score.data.cpu().numpy()
            print(score_list.shape)
            print(pred_mask_all.shape)
            savepath = './result/' + self.opt.LSN.filename
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            if not os.path.exists(savepath + "/mask"):
                os.makedirs(savepath + "/mask")
            if not os.path.exists(savepath + "/bbox"):
                os.makedirs(savepath + "/bbox")
            if not os.path.exists(savepath + "/score"):
                os.makedirs(savepath + "/score")
            np.save(savepath + "/mask/" + self.filename.split('.')[0] + "_mask.npy", pred_mask_all)
            np.save(savepath + "/bbox/" + self.filename.split('.')[0] + "_bbox.npy", bbox.data.cpu().numpy())
            np.save(savepath + "/score/" + self.filename.split('.')[0] + "_score.npy", score_list)

            return loss

    def res2json(self):
        pass