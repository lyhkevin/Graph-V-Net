# dataset and dataloader for pretraining
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image, ImageOps
import torch
import utils
import torchvision.transforms as T
from PIL import ImageEnhance, Image, ImageOps
from skimage.color import rgb2hed, hed2rgb
import random
import os
from glob import glob

class Pretrain_Dataset(data.Dataset):

    def __init__(self, opt, transform):
        self.img_transform = transform
        self.data_root = opt.data_path
        self.train_samples = opt.train_samples
        self.normal_path = []
        self.benign_path = []
        self.insitu_path = []
        self.invasive_path = []
        for subject_id in self.train_samples:
            self.normal_path = self.normal_path + glob(self.data_root + '/labeled/' + subject_id + '/*_0.png', recursive=True)
            self.benign_path = self.benign_path + glob(self.data_root + '/labeled/' + subject_id + '/*_1.png', recursive=True)
            self.insitu_path = self.insitu_path + glob(self.data_root + '/labeled/' + subject_id + '/*_2.png', recursive=True)
            self.invasive_path = self.invasive_path + glob(self.data_root + '/labeled/' + subject_id + '/*_3.png', recursive=True)
        self.unlabeled_path = glob(self.data_root + '/unlabeled/**/*.png', recursive=True)
        print(f"There are {len(self.normal_path) + len(self.benign_path) + len(self.insitu_path) + len(self.invasive_path)} labeled images, and {len(self.unlabeled_path)} unlabeled images.")
        print(f"Number of normal patches: {len(self.normal_path)}")
        print(f"Number of benign patches: {len(self.benign_path)}")
        print(f"Number of insitu patches: {len(self.insitu_path)}")
        print(f"Number of invasive patches: {len(self.invasive_path)}")
        self.img_path = self.normal_path + self.benign_path + self.insitu_path + self.invasive_path + self.unlabeled_path
        self.label = [0] * len(self.normal_path) + [1] * len(self.benign_path) + [2] * len(self.insitu_path) + [3] * len(self.invasive_path) + [-1] * len(self.unlabeled_path)
        self.img_transform = transform

    def __getitem__(self, index):
        imgs = []
        labels = []
        img_path = self.img_path[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        samples = self.img_transform(img)
        for sample in samples:
            imgs.append(sample)
            labels.append(self.label[index])
        return imgs, labels

    def __len__(self):
        return len(self.img_path)
