# -*- coding:utf-8 -*-
from engine.trainer import Trainer
from engine.env import Env
from data.build import build_dataloader
# from data.getdataloader import getDataLoader


class GRCNN_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data, test=False):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据
        '''
        from torch.autograd import Variable
        cpu_images, cpu_gt = data
        v_images = Variable(cpu_images.cuda())
        return (v_images,)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''
        from torch.autograd import Variable
        import torch
        if test == False:
            cpu_images, cpu_gt = originData
            text, text_len = self.converter.encode(cpu_gt)
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)
            bsz = cpu_images.size(0)
            predict_len = Variable(torch.IntTensor([modelResult.size(0)] * bsz))
            cost = self.criterion(modelResult, v_gt, predict_len, v_gt_len)
            return cost

        else:
            cpu_images, cpu_gt = originData
            bsz = cpu_images.size(0)
            text, text_len = self.converter.encode(cpu_gt)
            v_Images = Variable(cpu_images.cuda())
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)

            predict = modelResult
            # modelResult = self.model(v_Images)
            predict_len = Variable(torch.IntTensor([modelResult.size(0)] * bsz))
            cost = self.criterion(predict, v_gt, predict_len, v_gt_len)

            _, acc = predict.max(2)
            acc = acc.transpose(1, 0).contiguous().view(-1)

            sim_preds = self.converter.decode(acc.data, predict_len.data)

            return cost, sim_preds, cpu_gt

    def getScheduler(self):
        '''动态调整lr'''
        from torch.optim.lr_scheduler import LambdaLR, StepLR
        # return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        return StepLR(self.optimizer, step_size=20, gamma=0.1)


class TextSnake_Trainer(Trainer):

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)
        global val_result
        val_result = dict()

    def to_device(self, *tensors):
        return (t.to(self.opt.BASE.DEVICE) for t in tensors)

    def pretreatment(self, data, test=False):


        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = data
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        return (img,)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        if test == False:
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = originData
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                self.criterion(modelResult, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
            return loss

        else:
            import numpy as np
            import cv2
            import json
            from PIL import Image
            from model.detection_model.TextSnake_pytorch.util.detection import TextDetector
            from model.detection_model.TextSnake_pytorch.util.visualize import visualize_detection

            Image.MAX_IMAGE_PIXELS = None
            result = dict()
            detector = TextDetector(tcl_conf_thresh=0.3, tr_conf_thresh=1.0)
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = originData
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            self.criterion(modelResult, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
            for idx in range(img.size(0)):
                # print('detect {} / {} images: {}.'.format(i, len(test_loader), meta['image_id'][idx]))
                tr_pred = modelResult[idx, 0:2].softmax(dim=0).data.cpu().numpy()
                tcl_pred = modelResult[idx, 2:4].softmax(dim=0).data.cpu().numpy()
                sin_pred = modelResult[idx, 4].data.cpu().numpy()
                cos_pred = modelResult[idx, 5].data.cpu().numpy()
                radii_pred = modelResult[idx, 6].data.cpu().numpy()

                # tr_pred_mask = 1 / (1 + np.exp(-12*tr_pred[1]+3))
                tr_pred_mask = np.where(tr_pred[1] > detector.tr_conf_thresh, 1, tr_pred[1])
                # tr_pred_mask = fill_hole(tr_pred_mask)

                tcl_pred_mask = (tcl_pred * tr_pred_mask)[1] > detector.tcl_conf_thresh

                batch_result = detector.complete_detect(tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred,
                                                        radii_pred)  # (n_tcl, 3)
                # visualization
                img_show = img[idx].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * self.opt.TEXTSNAKE.stds + self.opt.TEXTSNAKE.means) * 255).astype(np.uint8)
                H, W = meta['Height'][idx].item(), meta['Width'][idx].item()

                # get pred_contours
                contours = []
                for instance in batch_result:
                    mask = np.zeros(img_show.shape[:2], dtype=np.uint8)
                    for disk in instance:
                        for x, y, r in disk:
                            cv2.circle(mask, (int(x), int(y)), int(r), (1), -1)

                    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(cont) > 0:
                        # for item in cont:
                        #     conts.append(item)
                        contours.append(cont[0])

                contours = [cont[:, 0, :] for cont in contours]

                polygons = []
                for cnt in contours:
                    drawing = np.zeros(tr_pred.shape[1:], np.int8)
                    mask = cv2.fillPoly(drawing, [cnt.astype(np.int32)], 1)
                    area = np.sum(np.greater(mask, 0))
                    if not area > 0:
                        continue

                    confidence = np.sum(mask * tr_pred[0]) / area

                    polygon = {
                        'points': cnt,
                        'confidence': confidence
                    }

                    polygons.append(polygon)

                h, w = img_show.shape[:2]
                # get no-padding image size
                resize_h = H if H % 32 == 0 else (H // 32) * 32
                resize_w = W if W % 32 == 0 else (W // 32) * 32
                ratio = float(h) / resize_h if resize_h > resize_w else float(w) / resize_w
                resize_h = int(resize_h * ratio)
                resize_w = int(resize_w * ratio)

                # crop no-padding image
                no_padding_image = img_show[0:resize_h, 0:resize_w, ::-1]
                no_padding_image = cv2.resize(no_padding_image, (W, H))

                # rescale points
                for polygon in polygons:
                    polygon['points'][:, 0] = (polygon['points'][:, 0] * float(W) / resize_w).astype(np.int32)
                    polygon['points'][:, 1] = (polygon['points'][:, 1] * float(H) / resize_h).astype(np.int32)

                img_show = no_padding_image

                # filter too small polygon
                for i, poly in enumerate(polygons):
                    if cv2.contourArea(poly['points']) < 100:
                        polygons[i] = []
                polygons = [item for item in polygons if item != []]

                # convert np.array to list
                for polygon in polygons:
                    polygon['points'] = polygon['points'].tolist()

                result[meta['image_id'][idx].replace('.jpg', '').replace('gt', 'res')] = polygons

            val_result.update(result)


                # print("Output json file in {}.".format(self.opt.ADDRESS.OUTPUT_DIR))
            return loss

    def res2json(self):
        import os
        global val_result
        val_result = dict()

        result_dir = self.opt.ADDRESS.RESULT_DIR
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        jpath = os.path.join(result_dir, 'result.json')
        if os.path.isfile(jpath):
            os.remove(jpath)
        return jpath

class MORAN_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data, test=False):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据
        '''
        import torch
        from torch.autograd import Variable
        from utils.loadData import loadData

        image = torch.FloatTensor(self.opt.MODEL.BATCH_SIZE, self.opt.IMAGE.IMG_CHANNEL, self.opt.IMAGE.IMG_H,
                                  self.opt.IMAGE.IMG_H)
        text = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        text_rev = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        length = torch.IntTensor(self.opt.MODEL.BATCH_SIZE)

        if self.opt.CUDA:
            # self.model = torch.nn.DataParallel(self.model, device_ids=range(self.opt.ngpu))
            image = image.cuda()
            text = text.cuda()
            text_rev = text_rev.cuda()
            self.criterion = self.criterion.cuda()

        image = Variable(image)
        text = Variable(text)
        text_rev = Variable(text_rev)
        length = Variable(length)

        if self.opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            t_rev, _ = self.converter.encode(cpu_texts_rev, scanned=True)
            loadData(text, t)
            loadData(text_rev, t_rev)
            loadData(length, l)
            return image, length, text, text_rev, test
            # preds0, preds1 = self.model(image, length, text, text_rev)
            # cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        else:
            cpu_images, cpu_texts = data
            loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            loadData(text, t)
            loadData(length, l)
            return image, length, text, text_rev, test
            # preds = self.model(image, length, text, text_rev)
            # cost = self.criterion(preds, text)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''
        import torch
        from torch.autograd import Variable

        if test == False:
            if self.opt.BidirDecoder:
                image, length, text, text_rev, _ = pretreatmentData
                preds0, preds1 = modelResult
                cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
            else:
                image, length, text, text_rev = pretreatmentData
                preds = self.model(image, length, text, text_rev)
                cost = self.criterion(preds, text)

            return cost

        else:
            if self.opt.BidirDecoder:
                preds0, preds1 = modelResult
                cpu_images, cpu_texts, cpu_texts_rev = originData
                image, length, text, text_rev, _ = pretreatmentData

                cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
                preds0, preds1 = modelResult
                preds0_prob, preds0 = preds0.max(1)
                preds0 = preds0.view(-1)
                preds0_prob = preds0_prob.view(-1)
                sim_preds0 = self.converter.decode(preds0.data, length.data)
                preds1_prob, preds1 = preds1.max(1)
                preds1 = preds1.view(-1)
                preds1_prob = preds1_prob.view(-1)
                sim_preds1 = self.converter.decode(preds1.data, length.data)
                sim_preds = []
                for j in range(cpu_images.size(0)):
                    text_begin = 0 if j == 0 else length.data[:j].sum()
                    if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).data[0] > \
                            torch.mean(
                                preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).data[0]:
                        sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                    else:
                        sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')

                return cost, sim_preds, cpu_texts
            else:
                cpu_images, cpu_texts = originData
                preds = modelResult
                image, length, text, text_rev, _ = pretreatmentData
                cost = self.criterion(preds, text)
                _, preds = preds.max(1)
                preds = preds.view(-1)
                sim_preds = self.converter.decode(preds.data, length.data)

                return cost, sim_preds, cpu_texts

    def getScheduler(self):
        from torch.optim.lr_scheduler import LambdaLR, StepLR
        return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

class AEAST_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data, test=False):
        img, gt = data
        img = img.cuda()
        gt = gt.cuda()
        return img, gt

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        img, gt = pretreatmentData
        if test == True:
            loss = self.criterion(gt, modelResult)
            return loss
        else:
            loss = self.criterion(gt, modelResult)
            return loss



env = Env()
train_loader, test_loader = build_dataloader(env.opt)
newTrainer = MORAN_Trainer(modelObject=env.model, opt=env.opt, train_loader=train_loader, val_loader=test_loader).train()
