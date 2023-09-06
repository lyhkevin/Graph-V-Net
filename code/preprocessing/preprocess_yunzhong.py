from glob import glob
import os
from code.preprocessing.filter import *
import openslide
import numpy as np
from tqdm import tqdm
import math
import cv2
import PIL.Image as Image

class yunzhong_info:
    def __init__(self):
        self.svs_path_all = glob('F:/raw/**/*.svs', recursive=True)
        self.base_dir = 'F:/large_patches/'
        os.makedirs(self.base_dir, exist_ok=True)

        self.subject_dir = None
        self.svs = None
        self.slide = None
        self.mask = None
        self.svs_name = None

        self.w_1x = None #the resolution of the WSI in 1x magnification
        self.h_1x = None

        self.threshold = 75 # tissue_rate threshold (lower than the threshold: discard the patch)
        self.patch_w_1x = 96 # patch size in 1x magnification
        self.patch_h_1x = 96
        self.patch_w_40x = self.patch_w_1x * 16 # patch size in 40x magnification
        self.patch_h_40x = self.patch_w_1x * 16
        self.img_size = (512, 512)  # resize the cropped patch in 40x magnification
        self.max_patch_num = 200 # the maximum number of patches in each WSI

    def makedir(self):
        self.subject_dir = self.base_dir + self.svs_name + '/'
        self.patch_dir = self.base_dir + self.svs_name + '/'
        os.makedirs(self.patch_dir, exist_ok=True)

def read_wsi(info):
    print("load svs...........")
    print('svs path:', info.svs_path)
    info.slide = openslide.OpenSlide(info.svs_path)
    info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    #thumbnail.save(info.subject_dir + 'thumbnail.png')
    slide = np.asarray(thumbnail)
    info.slide_np = np.clip(slide, 0, 255).astype(np.uint8)

def grid_with_num(info):
    grid_num = info.slide_np.copy()
    grid_num = cv2.resize(grid_num, dsize=(info.w_1x, info.h_1x), interpolation=cv2.INTER_CUBIC)
    for col_id in range(0, info.num_col_patch):
        for row_id in range(0, info.num_row_patch):
            x = row_id * info.patch_w_1x
            y = col_id * info.patch_h_1x
            cv2.rectangle(grid_num, (y, x), (y + 2 * info.patch_w_1x, x + 2 * info.patch_h_1x), (220, 220, 220), thickness=2)
            cv2.putText(grid_num, str(row_id), (y + 3, x + 13), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                        color=(128, 128, 128), thickness=1)
            cv2.putText(grid_num, str(col_id), (y + 3, x + 27), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                        color=(128, 128, 128), thickness=1)
    grid_num = Image.fromarray(grid_num)
    grid_num.save(info.subject_dir + 'grid.png')

def get_patch(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.patch_w_40x, info.patch_h_40x))
    patch = patch.resize(info.img_size)
    return patch

def otsu_threshold(info):
    print('perform otsu threshold..................')
    grayscale = filter_rgb_to_grayscale(info.slide_np)
    complement = filter_complement(grayscale)
    filtered = filter_otsu_threshold(complement)
    mask = Image.fromarray(filtered)
    #mask.save(info.subject_dir + 'mask.png')
    info.mask = np.clip(filtered, 0, 1)
    return

def get_tissue_rate(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    mask = info.mask[x:x+info.patch_w_1x,y:y+info.patch_h_1x]
    tissue_rate = int(np.sum(mask) * 100.0 / (mask.shape[0] * mask.shape[1]))
    return tissue_rate

def grid_wsi(info):
    print("grid and select patch from whole slide image................")
    info.num_row_patch = math.floor(info.h_1x / info.patch_h_1x)
    info.num_col_patch = math.floor(info.w_1x / info.patch_w_1x)
    print('num row:', info.num_row_patch, 'num col:', info.num_col_patch)
    #grid_with_num(info)
    count = 0
    for col_id in range(4, info.num_col_patch - 4, 2):
        for row_id in range(4, info.num_row_patch - 4, 2):
            tissue_rate = get_tissue_rate(row_id, col_id, info)
            if tissue_rate > info.threshold:
                patch = get_patch(row_id, col_id, info)
                patch.save(info.patch_dir + str(row_id) + '_' + str(col_id) + '.png')
                count = count + 1
                if count >= info.max_patch_num:
                    return

if __name__ == '__main__':
    info = yunzhong_info()
    for svs_path in tqdm(info.svs_path_all):
        info.svs_path = svs_path
        info.svs_name = os.path.splitext(os.path.split(svs_path)[-1])[0]
        print('preprocessing whole slide image:', info.svs_name)
        info.makedir()
        read_wsi(info)
        otsu_threshold(info)
        grid_wsi(info)