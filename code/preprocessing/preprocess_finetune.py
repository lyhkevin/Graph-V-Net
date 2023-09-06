from xml.etree.ElementTree import parse
from glob import glob
from torchvision.utils import draw_segmentation_masks as draw_mask
import torchvision.transforms.functional as F
import os
import torch
import tqdm
import openslide
import numpy as np
from tqdm import tqdm
import math
import cv2
import PIL.Image as Image

class bach_info:
    def __init__(self):
        self.svs_path_all = glob(r'../../dataset/ICIAR2018_BACH_Challenge/WSI/*.svs') + glob(r'../../dataset/ICIAR2018_BACH_Challenge_TestDataset/WSI/*.svs')
        self.xml_root = '../../dataset/annotations/'
        self.base_dir = '../../dataset/fine_tune/'
        os.makedirs(self.base_dir, exist_ok=True)
        self.subject_dir = None
        self.with_annotation = None
        self.svs = None
        self.slide = None
        self.slide_np = None
        self.mask = None
        self.w_1x = None # resolution of the WSI in 1x magnification
        self.h_1x = None

        self.stride = 4 # stride of the sliding window (in units of patches)
        self.patch_w_1x = 32 # patch size in 1x magnification
        self.patch_h_1x = 32
        self.patch_num = 8  # a cropped region consists of 8*8 patches
        self.region_w_40x = self.patch_w_1x * 16 * self.patch_num #region size in 40x magnification
        self.region_h_40x = self.patch_w_1x * 16 * self.patch_num
        self.patch_pix_count = self.patch_w_1x * self.patch_h_1x
        self.img_size = (224 * self.patch_num, 224 * self.patch_num) # resize the cropped region

    def makedir(self, svs_path, xml_path):
        self.svs_path = svs_path
        self.xml_path = xml_path
        self.name = self.svs_path.split('\\')[1][:-4]
        color_others, color_benign, color_insitu, color_invasive = (220, 220, 220), (0, 255, 0), (255, 255, 0), (255, 51, 0)
        self.Color = [color_others, color_benign, color_insitu, color_invasive]  # gray, green, yellow, red

        self.subject_dir = self.base_dir + self.name + '/'
        self.patch_dir = self.base_dir + self.name + '/' + 'region' + '/'
        self.label_dir = self.base_dir + self.name + '/' + 'label' + '/'

        os.makedirs(self.subject_dir, exist_ok=True)
        os.makedirs(self.patch_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

def read_wsi(info):

    print("load svs...........")
    print('svs path:', info.svs_path)
    info.slide = openslide.OpenSlide(info.svs_path)
    info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    thumbnail.save(info.subject_dir + 'thumbnail.png')
    slide = np.asarray(thumbnail)
    info.slide_np = np.clip(slide, 0, 255).astype(np.uint8)

def read_annotation(info):
    mask = np.zeros([info.h_1x, info.w_1x], dtype=np.uint8)
    xml = parse(info.xml_path).getroot()
    img_contour = info.slide_np.copy()
    for type in xml.iter('Annotation'):
        label = int(type.attrib.get('Id'))
        for region in type.iter('Region'):
            regions = []
            for vertex in region.iter('Vertex'):
                x, y = round(float(vertex.get('X')) / 16), round(float(vertex.get('Y')) / 16)
                regions.append([x, y])
            cv2.drawContours(mask, np.array([regions]), -1, label, thickness=cv2.FILLED)
            cv2.drawContours(img_contour, np.array([regions]), -1, info.Color[label], thickness=3)

    img_contour = Image.fromarray(img_contour)
    # draw overlay mask
    mask_bool = (np.arange(mask.max()) == mask[..., None] - 1).astype(bool)
    mask_tensor = torch.from_numpy(mask_bool)
    wsi_tensor = torch.from_numpy(info.slide_np)
    wsi_tensor = torch.permute(wsi_tensor, (2, 0, 1))
    mask_tensor = torch.permute(mask_tensor, (2, 0, 1))
    wsi_mask = draw_mask(wsi_tensor, mask_tensor, alpha=0.35, colors=info.Color[1:])
    wsi_mask = F.to_pil_image(wsi_mask)
    wsi_mask.save(info.subject_dir + 'annotation_mask.png')
    img_contour.save(info.subject_dir + 'contour.png')
    info.pix_annotation = mask
    np.save(info.subject_dir + 'pix_annotation.npy', info.pix_annotation)
    np.save(info.subject_dir + 'thumbnail.npy', info.slide_np)

def get_soft_label(info, patch):
    count = [0, 0, 0, 0]
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            count[patch[i][j]] += 1
    for i in range(len(count)):
        count[i] = (count[i] * 1.0) / info.patch_pix_count
    return np.array(count)

def get_patch_annotation(info): # get soft label for each patch
    info.patch_annotation = np.zeros((info.num_row_patch, info.num_col_patch, 4))
    for row_id in range(info.num_row_patch):
        for col_id in range(info.num_col_patch):
            x = row_id * info.patch_w_1x
            y = col_id * info.patch_h_1x
            label = get_soft_label(info, info.pix_annotation[x:x + info.patch_w_1x, y:y + info.patch_h_1x])
            info.patch_annotation[row_id][col_id] = label
    np.save(info.subject_dir + 'patch_annotation.npy', info.patch_annotation)

def grid_with_num(info):
    grid_num = info.slide_np.copy()
    grid_num = cv2.resize(grid_num, dsize=(info.w_1x, info.h_1x), interpolation=cv2.INTER_CUBIC)
    for row_id in range(info.num_row_patch):
        for col_id in range(info.num_col_patch):
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
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.region_w_40x, info.region_h_40x))
    patch = patch.resize(info.img_size)
    return patch

def grid_wsi(info):
    print("grid and select patch from whole slide image................")
    info.num_row_patch = math.floor(info.h_1x / info.patch_h_1x)
    info.num_col_patch = math.floor(info.w_1x / info.patch_w_1x)
    get_patch_annotation(info)
    print('num row:', info.num_row_patch, 'num col:', info.num_col_patch)
    grid_with_num(info)
    for row_id in tqdm(range(0, info.num_row_patch, info.stride)):
        for col_id in range(0, info.num_col_patch, info.stride):
            if row_id + info.patch_num < info.num_row_patch and col_id + info.patch_num < info.num_col_patch:
                patch = get_patch(row_id, col_id, info)
                label = info.patch_annotation[row_id:row_id + info.patch_num, col_id:col_id + info.patch_num]
                patch.save(info.patch_dir + str(row_id) + '_' + str(col_id) + '.png')
                np.save(info.label_dir + str(row_id) + '_' + str(col_id), label)
            else:
                row, col = row_id, col_id
                if row_id + info.patch_num >= info.num_row_patch:
                    row = info.num_row_patch - info.patch_num
                if col_id + info.patch_num >= info.num_col_patch:
                    col = info.num_col_patch - info.patch_num
                patch = get_patch(row, col, info)
                label = info.patch_annotation[row:row + info.patch_num, col:col + info.patch_num]
                patch.save(info.patch_dir + str(row) + '_' + str(col) + '.png')
                np.save(info.label_dir + str(row) + '_' + str(col), label)

if __name__ == '__main__':
    info = bach_info()
    for svs_path in info.svs_path_all:
        name = os.path.splitext(os.path.split(svs_path)[-1])[0]
        print('preprocessing whole slide image:', name)
        xml_path = info.xml_root + name + '.xml'
        if os.path.exists(xml_path):
            info.makedir(svs_path, xml_path)
            read_wsi(info)
            read_annotation(info)
            # grid_wsi(info)