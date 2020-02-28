import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossSigmoid(nn.Module):
    """
    sigmoid version focal loss
    """
    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets, train_mask):
        # training mask
        inputs = inputs[train_mask]
        targets = targets[train_mask]

        P = torch.sigmoid(inputs)
        zeros = P.new(P.size()).fill_(0.).float()
        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = torch.where(targets > zeros, targets - P, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = torch.where(targets > zeros, zeros, P)

        # clip norm for more stable
        eps = 1e-5
        P = torch.clamp(P, eps, 1-eps)

        loss_pos = - self.alpha * (pos_p_sub ** self.gamma) * torch.log(P)
        loss_neg = - (1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(1-P)
        batch_loss = loss_pos + loss_neg

        return batch_loss.sum()


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLossSigmoid(alpha=0.25, gamma=2, size_average=True)

    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()

        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))

        loss_pos = F.cross_entropy(predict[pos], target[pos], size_average=False)
        loss_neg = F.cross_entropy(predict[neg], target[neg], reduce=False)
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, tr_mask, tcl_mask, sin_map, cos_map, radii_map, train_mask):
        """
        calculate textsnake loss
        Args:
            input: (Variable), network predict, (BS, 7, H, W)
            tr_mask: (Variable), TR target, (BS, H, W)
            tcl_mask: (Variable), TCL target, (BS, H, W)
            sin_map: (Variable), sin target, (BS, H, W)
            cos_map: (Variable), cos target, (BS, H, W)
            radii_map: (Variable), radius target, (BS, H, W)
            train_mask: (Variable), training mask, (BS, H, W)

        Returns:
            loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """

        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        sin_pred = input[:, 4].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = input[:, 5].contiguous().view(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        radii_pred = input[:, 6].contiguous().view(-1)  # (BSxHxW,)
        train_mask = train_mask.view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        radii_map = radii_map.contiguous().view(-1)
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)

        # # loss_tr = F.cross_entropy(tr_pred[train_mask], tr_mask[train_mask].long())
        # loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        # loss_tcl = F.cross_entropy(tcl_pred[train_mask * tr_mask], tcl_mask[train_mask * tr_mask].long())
        # focal_loss
        tr_focal_mask = torch.cat(((1-tr_mask.long()).unsqueeze(1).float(), tr_mask.long().unsqueeze(1).float()), dim=1)
        loss_tr = self.focal_loss(tr_pred, tr_focal_mask, train_mask.byte()) * 1e-5

        # tcl_mask = torch.cat(((1-tcl_mask).unsqueeze(1).float(), tcl_mask.unsqueeze(1).float()), dim=1)
        # loss_tcl = self.focal_loss(tcl_pred, tcl_mask, (train_mask.byte() * tr_mask[:, 1].byte())) * 1e-5
        loss_tcl = F.cross_entropy(tcl_pred[train_mask * tr_mask], tcl_mask[train_mask * tr_mask].long())

        # geometry losses
        # tcl_mask = tcl_mask[:, 1].byte()
        ones = radii_map.new(radii_pred[tcl_mask].size()).fill_(1.).float()
        loss_radii = F.smooth_l1_loss(radii_pred[tcl_mask] / radii_map[tcl_mask], ones)
        loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
        loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        return loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
