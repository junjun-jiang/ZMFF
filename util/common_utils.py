import os
import PIL
import sys
import cv2
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from thop import profile
from util.guided_filter import guided_filter


def label_sample():
    label = torch.LongTensor(1, 1).random_() % 1000
    # scatter_(dim, index, src, reduce=None)
    # self[index[i]]=src
    one_hot = torch.zeros(1, 1000).scatter_(1, label, 1)
    return label.squeeze(1).cuda(), one_hot.cuda()


def get_score_map(y1, y2, mode='blur2th'):
    score_map_ = torch.sign(torch.abs(blur_2th(y1)) - torch.min(torch.abs(blur_2th(y1)), torch.abs(blur_2th(y2))))
    if mode == 'blur2th':
        score_map = score_map_
    elif mode == 'max_select':
        score_map = torch.max(torch.abs(blur_2th(y1)), torch.abs(blur_2th(y2)))
    elif mode == 'gradient':
        score_map = torch.sign(torch.abs(gradient(y1)) - torch.min(torch.abs(gradient(y1)), torch.abs(gradient(y2))))
    elif mode == 'guassian':
        score_map = guassian(score_map_)
    elif mode == 'guided_filter':
        score_map = torch.from_numpy(guided_filter(I=y1.squeeze().cpu().numpy(), p=score_map_.squeeze().cpu().numpy(), r=8, eps=0.05)).unsqueeze(0).unsqueeze(0)
    else:
        raise NotImplementedError
    return score_map.cuda()


def blur_2th(img):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]], device=img.device)
    assert img.ndim == 4 and (img.shape[1] == 1 or img.shape[1] == 3)
    filtr = filtr.expand(img.shape[1], img.shape[1], 3, 3)
    blur = F.conv2d(img, filtr, bias=None, stride=1, padding=1)
    blur = F.conv2d(blur, filtr, bias=None, stride=1, padding=1)
    diff = torch.abs(img - blur)
    return diff

def guassian(input1):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]).type(torch.cuda.FloatTensor)
    filtr = filtr.expand(input1.shape[1], input1.shape[1], 3, 3)
    blur = F.conv2d(input1, filtr, bias=None, stride=1, padding=1)
    return blur


def gradient(input1):
    n, c, w, h = input1.shape
    filter1 = torch.reshape(torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).type(torch.cuda.FloatTensor), [1, 1, 3, 3])
    filter1 = filter1.repeat_interleave(c, dim=1)
    d = torch.nn.functional.conv2d(input1, filter1, bias=None, stride=1, padding=1)
    return d


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''
    imgsize = img.shape

    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)

    bbox = [
        int((imgsize[0] - new_size[0]) / 2),
        int((imgsize[1] - new_size[1]) / 2),
        int((imgsize[0] + new_size[0]) / 2),
        int((imgsize[1] + new_size[1]) / 2),
    ]

    img_cropped = img[0:new_size[0], 0:new_size[1], :]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    # plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    # if images_np[0].shape[0] == 1:
    #    plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    # else:
    #    plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    # plt.axis('off')
    # plt.show()
    
    return grid


def load(path, channel):
    """Load PIL image."""
    if channel == 1:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path)

    return img


def get_image(path, imsize=-1, channel=1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path, channel)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1:
        img = img.resize(imsize)

    # if imsize[0]!= -1 and img.size != imsize:
    #     if imsize[0] > img.size[0]:
    #         img = img.resize(imsize, Image.BICUBIC)
    #     else:
    #         img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)
    img = np_to_torch(img_np)

    return img


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(spatial_size, input_channel, input_type='noise', noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if input_type == 'noise':
        shape = [1, input_channel, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif input_type == 'meshgrid':
        assert input_channel == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid).type(torch.cuda.FloatTensor)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    if len(img_np.shape) == 3:
        return torch.from_numpy(img_np)[None, :]
    else:
        return torch.from_numpy(img_np)[None, None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def read_img(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    y = np_to_torch((y / 255.).astype(np.float32)).cuda()
    img = np_to_torch((img.transpose(2, 0, 1) / 255.).astype(np.float32)).cuda()

    return img, y, cb, cr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def label_sampel():
    label = torch.LongTensor(1, 1).random_() % 1000
    # scatter_(dim, index, src, reduce=None)
    # self[index[i]]=src
    one_hot = torch.zeros(1, 1000).scatter_(1, label, 1)
    return label.squeeze(1).cuda(), one_hot.cuda()


