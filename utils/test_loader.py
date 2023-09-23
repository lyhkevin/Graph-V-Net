# dataset and dataloader for pretraining
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from einops import rearrange
from PIL import ImageEnhance, Image, ImageOps
from skimage.color import rgb2hed, hed2rgb
import random
import os
import glob
import torch
import glob
import logging

class Test_Dataset(data.Dataset):

    def __init__(self, opt):
        self.data_root = opt.data_root
        self.opt = opt
        self.subject_id = opt.subject_id
        self.subject_path = self.data_root + opt.subject_id + '/'
        print('subject path:', self.subject_path)

        self.label = ['normal', 'benign', 'insitu', 'invasive']
        self.img_size = opt.patch_size * 8
        self.patch_size = opt.patch_size
        self.img_path, self.label_path = self.get_path(self.subject_path)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(opt.img_size)
        ])
        self.thumbnail = np.load(self.subject_path + '/thumbnail.npy')
        self.pix_annotation = np.load(self.subject_path + '/pix_annotation.npy')
        self.patch_annotation = np.load(self.subject_path + '/patch_annotation.npy')
        self.pix_prediction = np.zeros(self.pix_annotation.shape)
        x,y = self.patch_annotation.shape[0], self.patch_annotation.shape[1]
        self.patch_prediction = np.zeros((x,y,4))
        self.thumbnail_img = Image.open(self.subject_path + '/annotation_mask.png')
        print('row tiles:', self.patch_annotation.shape[0], 'col tiles:', self.patch_annotation.shape[1])
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_label(self):
        return self.thumbnail, self.pix_annotation, self.patch_annotation, self.pix_prediction, self.patch_prediction, self.thumbnail_img

    def get_path(self, path):

        img_root = path + '/region/'
        label_root = path + '/label/'
        img_path = sorted([img_root + f for f in os.listdir(img_root)])
        label_path = sorted([label_root + f for f in os.listdir(label_root)])

        return img_path, label_path

    def patchify(self, img):

        img = rearrange(img, 'c (x h) (y w) -> (x y) c h w', h=self.patch_size,
                        w=self.patch_size, x=8, y=8)
        return img

    def get_coord(self, path):
        coord = path.split('/')[-1][:-4].split('_')
        return int(coord[0]), int(coord[1])

    def __getitem__(self, index):

        img_path = self.img_path[index]
        coord = self.get_coord(img_path)

        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.img_transform(img)
        patches = self.patchify(img)

        return patches, coord

    def __len__(self):
        return len(self.img_path)

def get_test_loader(batch_size, shuffle, pin_memory, num_workers, opt):
    dataset = Test_Dataset(opt)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                  pin_memory=pin_memory)
    return dataset, data_loader