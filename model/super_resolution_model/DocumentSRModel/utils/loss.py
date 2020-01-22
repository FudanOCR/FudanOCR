import torch
import torch.nn.functional as F

def BCE2d(ipt, target):
    n, c, h, w = ipt.size()

    log_p = ipt.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight = torch.Tensor(log_p.size()).fill_(0).numpy()
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num

    weight = torch.from_numpy(weight).cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss