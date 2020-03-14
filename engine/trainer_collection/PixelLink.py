from engine.trainer import Trainer

class PixelLink_Trainer(Trainer):
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
        img = data['image']
        img = img.cuda()
        if test == True:
            return img
        else:
            return (img, )

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        # img, pixel_mask, neg_pixel_mask, gt, pixel_pos_weight, link_mask = originData
        # gt = originData['label']
        # gt = gt.cuda()
        pixel_masks = originData['pixel_mask'].cuda()
        neg_pixel_masks = originData['neg_pixel_mask'].cuda()
        link_masks = originData['link_mask'].cuda()
        pixel_pos_weights = originData['pixel_pos_weight'].cuda()

        out_1, out_2 = modelResult

        from engine.personalize_loss.pixellink_loss import PixelLinkLoss
        loss_instance = PixelLinkLoss(self.opt)

        pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(out_1, pixel_masks, neg_pixel_masks, pixel_pos_weights)
        pixel_loss = pixel_loss_pos + pixel_loss_neg
        link_loss_pos, link_loss_neg = loss_instance.link_loss(out_2, link_masks)
        link_loss = link_loss_pos + link_loss_neg
        losses = self.opt.MODEL.PIXEL_WEIGHT * pixel_loss + self.opt.MODEL.LINK_WEIGHT * link_loss
        # print("total loss: " + str(losses.tolist()), end=", ")
        return losses