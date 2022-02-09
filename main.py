# from __future__ import print_function

import os
import glob
import time
import yaml
import torch
import warnings
import argparse
import numpy as np

from tqdm import tqdm
from network.loss import Loss
from network.skip import skip
from util.logger import Logger
from util.common_utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from util.dataset import Lytro, Real_MFF, MFI_WHU, Lytro3

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/test.yaml')
    args = parser.parse_args()

    return args


def main(config, log, writer):

    ##################################################### dataset ######################################################

    if config['dataset'] == 'Lytro':
        test_set = Lytro(config['data_path'])
    elif config['dataset'] == 'Real-MFF':
        test_set = Real_MFF(config['data_path'])
    elif config['dataset'] == 'MFI-WHU':
        test_set = MFI_WHU(config['data_path'])
    elif config['dataset'] == 'Lytro3':
        test_set = Lytro3(config['data_path'])
    else:
        raise NotImplementedError

    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    loss = Loss(config, device)

    for (y1, y2, cr, cb, img_name) in loader:
        img_name = img_name[0]
        log.logger.info(img_name)
        y_list = [i.to(device) for i in [y1, y2]]

        n, c, h, w = y1.shape
        config['img_size'][0] = h
        config['img_size'][1] = w

        ################################################## network #####################################################

        net_inputx = torch.cat(y_list, dim=1).to(device)
        netx = skip(config['input_channelx'], 1,
                    channels=[128, 128, 128, 128, 128],
                    channels_skip=16,
                    upsample_mode='bilinear',
                    attention_mode=config['attention'],
                    need_bias=False, pad=config['pad'], act_fun='LeakyReLU', scales=config['scales']).to(device)

        net_inputm = []
        netm = []
        for i in range(config['num_source']):
            net_inputm.append(get_noise(spatial_size=config['img_size'],
                                        input_channel=config['input_channelm'],
                                        input_type=config['input_typem']).to(device))
            netm.append(skip(config['input_channelm'], 1,
                             channels=[128, 128, 128],
                             channels_skip=16,
                             upsample_mode='bilinear',
                             attention_mode=config['attention'],
                             need_bias=False, pad=config['pad'], act_fun='LeakyReLU', scales=config['scales']).to(device))

        # with torch.no_grad():
        #     for i in range(config['num_source']):
        #         for j in range(config['input_channelm']):
        #             save_path_n = os.path.join(config['save_path'] + config['experiment_name'],
        #                                        '%s_n%d.png' % (img_name, i*config['num_source']+j))
        #             cv2.imwrite(save_path_n, np.uint8(np.clip(torch_to_np(net_inputm[i][:, j])*4096, 0, 255).squeeze()))

        ############################################### optimizer ######################################################

        total_parameters = sum([param.nelement() for param in netx.parameters()])
        parameters = [{'params': netx.parameters()}, {'params': net_inputx}]
        for i in range(config['num_source']):
            parameters.extend([{'params': netm[i].parameters()}, {'params': net_inputm[i]}])
            total_parameters += sum([param.nelement() for param in netm[i].parameters()])

        optimizer = torch.optim.Adam(parameters, lr=config['lr'])

        # FLOPs1, params1 = profile(model=net1, inputs=net_input1)
        # FLOPs2, params2 = profile(model=net2, inputs=net_input2)
        #
        log.logger.info("Total parameters: %.2fM" % (total_parameters / 1e6))
        # log.logger.info("FLOPs: %.2fG" % ((FLOPs1+FLOPs2) / 1e9))
        # log.logger.info("params: %.2fM" % ((params1+params2) / 1e6))

        # using multi-step as the learning rate change strategy
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)  # learning rates

        net_input_savedx = net_inputx.detach().clone()
        noisex = net_inputx.detach().clone()
        net_input_savedm = []
        noisem = []
        max_map = torch.zeros_like(y_list[-1], device=device)
        for i in range(config['num_source']):
            net_input_savedm.append(net_inputm[i].detach().clone())
            noisem.append(net_inputm[i].detach().clone())
            max_map = torch.max(torch.abs(blur_2th(y_list[i].to(device))), max_map)

        score_map = []
        for i in range(config['num_source']):
            score_map.append(1 - torch.sign(max_map - torch.abs(blur_2th(y_list[i].to(device)))))

        ################################################ start iteration ###############################################

        for step in tqdm(range(config['num_iter'])):
            optimizer.zero_grad()

            # add_noise
            if config['reg_noise_std'] > 0:
                net_inputx = net_input_savedx + (noisex.normal_() * config['reg_noise_std'])
                for i in range(config['num_source']):
                    net_inputm[i] = net_input_savedm[i] + (noisem[i].normal_() * config['reg_noise_std'])

            # get the network output
            out_x = netx(net_inputx)
            out_x = [F.interpolate(i, size=config['img_size'], mode='bilinear', align_corners=False) for i in out_x]
            out_m = []
            for i in range(config['num_source']):
                out_m.append(netm[i](net_inputm[i]))
                out_m[i] = [F.interpolate(j, size=config['img_size'], mode='bilinear', align_corners=False) for j in out_m[i]]

            losses = loss(out_x, out_m, y_list, score_map, step)
            losses['total_loss'].backward()
            optimizer.step()

            # write to tensorboard
            writer.add_scalar(img_name + "/loss_recon", losses['recon_loss'], step)
            writer.add_scalar(img_name + "/loss_prior", losses['prior_loss'], step)
            writer.add_scalar(img_name + "/loss_total", losses['total_loss'], step)
            # writer.add_scalar(img_name + "/psnr", losses['psnr'], step)
            if step % config['percep_freq'] == 0:
                writer.add_scalar(img_name + "/loss_percep", losses['percep_loss'], step)

            # change the learning rate
            scheduler.step(step)

            ############################################### saving #####################################################

            if (step + 1) % config['save_freq'] == 0:
                with torch.no_grad():
                    # remove the padding area
                    # out_x_np = out_x_np[:, padh//2:padh//2+img_size_input[1], padw//2:padw//2+img_size_input[2]]

                    save_path_x = os.path.join(config['save_path']+config['experiment_name'], '%s_%d_x.png' % (img_name, step))
                    out_x_np = np.uint8(255 * torch_to_np(out_x[0]).squeeze())
                    out_x_np = cv2.merge([out_x_np, cr[0].squeeze().numpy(), cb[0].squeeze().numpy()])
                    out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(save_path_x, out_x_np)

                    for i in range(config['num_source']):
                        save_path_m = os.path.join(config['save_path']+config['experiment_name'],
                                                   '%s_%d_m%d.png' % (img_name, step, i))
                        cv2.imwrite(save_path_m, np.uint8(255 * torch_to_np(out_m[i][0]).squeeze()))

                    # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))

        for i in range(config['num_source']):
            save_path_s = os.path.join(config['save_path'] + config['experiment_name'], '%s_s%d.png' % (img_name, i))
            cv2.imwrite(save_path_s, np.uint8(255. * torch_to_np(score_map[i]).squeeze()))


if __name__ == '__main__':

    ################################################### preperation ####################################################

    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_random_seed(config['rand_seed'])

    os.makedirs(config['save_path']+config['experiment_name'], exist_ok=True)
    os.makedirs(config['writer_path']+config['experiment_name'], exist_ok=True)
    writer = SummaryWriter(config['writer_path']+config['experiment_name'])
    log = Logger(filename=config['save_path']+config['experiment_name']+'.log', level='debug')

    config_str = ""
    for k, v in config.items():
        config_str += '\n\t{:<30}: {}'.format(k, str(v))
    log.logger.info(config_str)

    main(config, log, writer)

    writer.close()
    log.logger.info("all done at %s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
