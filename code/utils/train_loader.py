from torchvision import transforms
import torch.utils.data as data
import numpy as np
from einops import rearrange
from PIL import ImageEnhance, Image, ImageOps
import random
import os
import torch

def random_flip(img, label):
    flip_flag = random.randint(0, 2)
    if flip_flag == 2:
        img = ImageOps.mirror(img)
        label = np.fliplr(label)
    return img, label

def randomRotation(image, label):
    rotate_time = random.randint(0, 3)
    image = image.rotate(rotate_time * 90)
    label = np.rot90(label, k=rotate_time, axes=(0, 1))
    return image, label

def colorEnhance(image):
    bright_intensity = random.randint(7, 13) / 10.0
    contrast_intensity = random.randint(7, 13) / 10.0
    color_intensity = random.randint(7, 13) / 10.0
    sharp_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    image = ImageEnhance.Color(image).enhance(color_intensity)
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return img

def randomPeper(img):
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 1
    return img


class Train_Dataset(data.Dataset):

    def __init__(self, opt):
        self.data_root = opt.data_root
        self.train_subject = opt.train_samples
        print('the IDs of patients for training:', self.train_subject)
        self.train_path = []
        self.label = ['normal', 'benign', 'insitu', 'invasive']
        self.patch_size = opt.patch_size
        self.img_size = opt.patch_size * 8 # a region consists of 8 * 8 patches
        self.aug = opt.augment
        self.oversample = opt.oversample
        self.img_train, self.label_train = self.get_path()

        print("[Training set Stats:] [Normal Num %d] [Benign Num: %d] [Insitu Num: %d] [Invasive Num: %d]"
              % (len(self.img_train[0]), len(self.img_train[1]), len(self.img_train[2]), len(self.img_train[3])))

        self.img_train[1] = self.img_train[1] * self.oversample[1]
        self.img_train[2] = self.img_train[2] * self.oversample[2]
        self.img_train[3] = self.img_train[3] * self.oversample[3]

        self.label_train[1] = self.label_train[1] * self.oversample[1]
        self.label_train[2] = self.label_train[2] * self.oversample[2]
        self.label_train[3] = self.label_train[3] * self.oversample[3]

        print("[After oversampled:] [Normal Num %d] [Benign Num: %d] [Insitu Num: %d] [Invasive Num: %d]"
          % (len(self.img_train[0]), len(self.img_train[1]), len(self.img_train[2]), len(self.img_train[3])))

        self.img_path = sum(self.img_train, [])
        self.label_path = sum(self.label_train, [])

        print('num samples:', len(self.img_path))

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size)
        ])
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_path(self):
        img_path = [[], [], [], []]
        label_path = [[], [], [], []]
        for subject in self.train_subject:
            img_root = self.data_root + subject + '/region/'
            label_root = self.data_root + subject + '/label/'
            img_root = sorted([img_root + f for f in os.listdir(img_root)])
            label_root = sorted([label_root + f for f in os.listdir(label_root)])
            for i, (img, label) in enumerate(zip(img_root, label_root)):
                label_npy = np.load(label) # (8, 8, 4), the patch-level label for 8*8 patches in a region
                label_npy = np.sum(label_npy, axis=(0, 1)) / 64.0
                normal_rate, benign_rate, insitu_rate, invasive_rate = label_npy[0], label_npy[1], label_npy[2], label_npy[3]
                if normal_rate == 1:
                    img_path[0].append(img)
                    label_path[0].append(label)
                else:
                    if benign_rate > 0:
                        img_path[1].append(img)
                        label_path[1].append(label)
                    if insitu_rate > 0:
                        img_path[2].append(img)
                        label_path[2].append(label)
                    if invasive_rate > 0:
                        img_path[3].append(img)
                        label_path[3].append(label)
        return img_path, label_path

    def patchify(self, img):
        img = rearrange(img, 'c (x h) (y w) -> (x y) c h w', h=self.patch_size, w=self.patch_size, x=8, y=8)
        return img

    def __getitem__(self, index):

        img_path = self.img_path[index]
        label_path = self.label_path[index]
        img = Image.open(img_path)
        label = np.load(label_path)
        img = img.convert('RGB')

        if self.aug == False:
            img = self.img_transform(img)
        else:
            img, label = random_flip(img, label)
            img, label = randomRotation(img, label)
            img = colorEnhance(img)
            img = self.img_transform(img)

        patches = self.patchify(img)
        label = torch.from_numpy(label.copy())
        return img, patches, label

    def __len__(self):
        return len(self.img_path)

def get_train_loader(batch_size, shuffle, pin_memory, num_workers, opt):
    dataset = Train_Dataset(opt)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return dataset, data_loader