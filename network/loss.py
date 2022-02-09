import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from network.SSIM import SSIM
from torch.nn import MSELoss, L1Loss
from skimage.measure import compare_psnr, compare_ssim


class Loss(nn.Module):
    def __init__(self, config, device):
        super(Loss, self).__init__()
        self.mse_loss = MSELoss().to(device)
        self.l1_loss = L1Loss().to(device)
        self.ssim_loss = SSIM().to(device)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.precep_freq = config['percep_freq']
        self.num_iter = config['num_iter']
        self.rate = config['rate']
        self.scales = config['scales']
        self.thresh = config['thresh']
        self.num_source = config['num_source']
        #
        # self.resnet = models.resnet101(pretrained=True).cuda()
        # for p in self.resnet.parameters():
        #     p.requires_grad = False
        # self.resnet.conv1.weight = torch.nn.Parameter(self.resnet.conv1.weight[:, :1, :, :])

    def forward(self, out_x, out_m, y_list, score_m, step):
        loss_percep = 0
        loss_recon = 0
        loss_prior = 0
        # if step % self.precep_freq == 0:
        #     out_x_percep = self.resnet(out_x[0]).cuda()
        #     y1_percep = self.resnet(y1).cuda()
        #     y2_percep = self.resnet(y2).cuda()
        #     loss_percep = self.l1_loss(out_x_percep, y1_percep) + self.l1_loss(out_x_percep, y2_percep)

        sum_score = torch.zeros_like(score_m[-1])
        for i in range(self.num_source):
            for j in range(self.scales):
                loss_prior += self.l1_loss(out_m[i][j], score_m[i])
            sum_score += score_m[i]
        loss_prior /= self.num_source * self.scales
        loss_prior += self.l1_loss(torch.ones_like(score_m[-1]), sum_score)

        gamma = [math.pow(self.rate, self.scales-1-i) for i in range(self.scales)]
        predicted = torch.zeros_like(y_list[-1], device=y_list[-1].device)
        if step < self.thresh:
            for i in range(self.scales):
                for j in range(self.num_source):
                    predicted += y_list[j]*out_m[j][i]
                loss_recon += gamma[i] * self.l1_loss(out_x[i], predicted)
        else:
            for i in range(self.scales):
                for j in range(self.num_source):
                    predicted += y_list[j] * out_m[j][i]
                loss_recon += gamma[i] * (1-self.ssim_loss(out_x[i], predicted))
        loss_recon /= sum(gamma)

        # loss_grad = l1(gradient(out_x), torch.max(gradient(y1),gradient(y2))
        # loss_prior1 = (torch.mean(torch.mul(score_map, torch.square(out_x - y1))) + torch.mean(torch.mul(
        #     1 - score_map, torch.square(out_x - y2)))).type(dtype)

        if step < self.thresh:
            total_loss = loss_recon + self.alpha * loss_prior + self.beta * loss_percep
        else:
            total_loss = loss_recon + self.beta * loss_percep

        # psnr = compare_psnr(np.uint8((out_m[0][-1]*y_list[0]+(1-out_m[0][-1])*y_list[1]).squeeze().detach().clone().cpu()*255),
        #                     np.uint8(gt.squeeze().detach().clone().cpu()*255))

        losses = {"total_loss": total_loss, "recon_loss": loss_recon, "prior_loss": loss_prior,
                  "percep_loss": loss_percep}
        return losses
