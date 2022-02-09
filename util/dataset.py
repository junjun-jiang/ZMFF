import os
import glob
import torch
import numpy as np

from util.common_utils import *
from torchvision import transforms
from torch.utils.data import Dataset


class Lytro(Dataset):
    def __init__(self, root_dir):
        self.images1 = glob.glob(os.path.join(root_dir, '*-A.jpg'))
        self.images1.sort()
        self.images2 = glob.glob(os.path.join(root_dir, '*-B.jpg'))
        self.images2.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.images1[idx])
        img2 = cv2.imread(self.images2[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)

        return y1, y2, cr, cb, os.path.basename(self.images1[idx]).split('.')[0][:-2]


class Real_MFF(Dataset):
    def __init__(self, root_dir):
        if not os.path.exists(os.path.join(root_dir, 'list2.txt')):
            files = os.listdir(root_dir)
        else:
            with open(os.path.join(root_dir, 'list2.txt'), 'r') as f:
                files = f.read().rstrip().split('\n')

        # files = files1[:int(len(files1) / 2)]
        # with open(os.path.join(root_dir, 'list1.txt'), 'w') as f:
        #     name = ''
        #     for file in files:
        #         name += file + '\n'
        #     f.write(name)
        # files = files1[int(len(files1) / 2):]
        # with open(os.path.join(root_dir, 'list2.txt'), 'w') as f:
        #     name = ''
        #     for file in files:
        #         name += file + '\n'
        #     f.write(name)

        files = ['IMG_0211']

        self.images1 = [os.path.join(root_dir, file, file + '_1.png') for file in files]
        self.images2 = [os.path.join(root_dir, file, file + '_2.png') for file in files]
        self.gts = [os.path.join(root_dir, file, file + '_0.png') for file in files]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.images1[idx])
        img2 = cv2.imread(self.images2[idx])
        gt = cv2.imread(self.gts[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb))
        gt, _, _ = cv2.split(cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)
        gt = self.transform(gt)

        return y1, y2, cr, cb, os.path.basename(self.images1[idx]).split('.')[0][:-2], gt


class MFI_WHU(Dataset):
    def __init__(self, root_dir):
        # [9, ]
        self.images1 = [os.path.join(root_dir, 'source_1', str(i)+'.jpg') for i in [9]]
        # self.images1.sort()
        self.images2 = [os.path.join(root_dir, 'source_2', str(i)+'.jpg') for i in [9]]
        # self.images2.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.images1[idx])
        img2 = cv2.imread(self.images2[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)

        return y1, y2, cr, cb, os.path.basename(self.images1[idx]).split('.')[0]


class Lytro3(Dataset):
    def __init__(self, root_dir):
        self.images1 = glob.glob(os.path.join(root_dir, 'Triple Series', '*-A.jpg'))
        self.images1.sort()
        self.images2 = glob.glob(os.path.join(root_dir, 'Triple Series', '*-B.jpg'))
        self.images2.sort()
        self.images3 = glob.glob(os.path.join(root_dir, 'Triple Series', '*-C.jpg'))
        self.images3.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.images1[idx])
        img2 = cv2.imread(self.images2[idx])
        img3 = cv2.imread(self.images3[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb))
        y3, _, _ = cv2.split(cv2.cvtColor(img3, cv2.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)
        y3 = self.transform(y3)

        return y1, y2, y3, cr, cb, os.path.basename(self.images1[idx]).split('.')[0][:-2]