import torch
import torch.nn as nn

import config as cfg


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        return

    def forward(self, y_true, y_pred):
        # loss of inside score
        logits = y_pred[:, :, :, :1]
        labels = y_true[:, :, :, :1]
        # balance positive and negative samples in an image
        beta = 1 - torch.mean(labels)
        # apply sigmoid activation
        predicts = torch.sigmoid(logits)
        # log + epsilon for stable cal
        inside_score_loss = torch.mean(-1 * (beta * labels * torch.log(predicts + cfg.epsilon) + (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
        inside_score_loss *= cfg.lambda_inside_score_loss

        # loss of side vertex code
        vertex_logits = y_pred[:, :, :, 1:3]
        vertex_labels = y_true[:, :, :, 1:3]
        vertex_beta = 1 - (torch.mean(y_true[:, :, :, 1:2]) / (torch.mean(labels) + cfg.epsilon))
        vertex_predicts = torch.sigmoid(vertex_logits)

        pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts + cfg.epsilon)
        neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(1 - vertex_predicts + cfg.epsilon)
        pos_weights = torch.eq(y_true[:, :, :, 0], torch.tensor(1.).cuda()).float()

        side_vertex_code_loss = torch.sum(torch.sum(pos + neg, dim=-1) * pos_weights) / (torch.sum(pos_weights) + cfg.epsilon)
        side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

        # loss of side vertex coord delta
        g_hat = y_pred[:, :, :, 3:]
        g_true = y_true[:, :, :, 3:]
        vertex_weights = torch.eq(y_true[:, :, :, 1], torch.tensor(1.).cuda()).float()
        pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)

        side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (torch.sum(vertex_weights) + cfg.epsilon)
        side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss

        return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss


def smooth_l1_loss(predict_tensor, target_tensor, weights):
    n_q = torch.reshape(quad_norm(target_tensor), weights.shape)
    diff = torch.abs((predict_tensor - target_tensor))
    diff_lt_1 = diff < 1
    pixel_wise_smooth_l1norm = (torch.sum(torch.where(diff_lt_1, 0.5 * torch.pow(diff, 2), diff - 0.5), dim=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    shape = g_true.shape
    delta_xy_matrix = torch.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = torch.pow(diff, 2)
    distance = torch.sqrt(torch.sum(square, dim=-1))
    distance *= 4.0
    distance += cfg.epsilon
    return torch.reshape(distance, shape[:-1])
