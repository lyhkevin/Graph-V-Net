from xml.etree.ElementTree import parse
from glob import glob
import tqdm
import os
from filter import *
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
        self.base_dir = '../../dataset/pretrain/'
        os.makedirs(self.base_dir, exist_ok=True)
        self.subject_dir = None
        self.with_annotation = None
        self.svs = None
        self.slide = None
        self.slide_np = None
        self.mask = None
        self.w_1x = None # the resolution of the WSI in 1x magnification
        self.h_1x = None
        self.threshold = 50 # tissue_rate threshold (lower than the threshold: discard the patch)
        self.patch_w_1x = 32 # patch size in 1x magnification
        self.patch_h_1x = 32
        self.patch_w_40x = self.patch_w_1x * 16 # patch size in 40x magnification
        self.patch_h_40x = self.patch_w_1x * 16
        self.img_size = (512, 512)  # resize the cropped patch in 40x magnification

    def makedir(self, svs_path, xml_path):
        self.svs_path = svs_path
        self.xml_path = xml_path
        self.name = self.svs_path.split('\\')[1][:-4]
        if self.with_annotation == True:
            self.subject_dir = self.base_dir + 'labeled/' + self.name + '/'
        else:
            self.subject_dir = self.base_dir + 'unlabeled/' + self.name + '/'
        os.makedirs(self.subject_dir, exist_ok=True)

def read_wsi(info):
    print("load svs...........")
    print('svs path:', info.svs_path)
    info.slide = openslide.OpenSlide(info.svs_path)
    info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    slide = np.asarray(thumbnail)
    info.slide_np = np.clip(slide, 0, 255).astype(np.uint8)

def read_annotation(info):
    if info.with_annotation == True:
        mask = np.zeros([info.h_1x, info.w_1x], dtype=np.uint8)
        xml = parse(info.xml_path).getroot()
        for type in xml.iter('Annotation'):
            label = int(type.attrib.get('Id'))
            for region in type.iter('Region'):
                regions = []
                for vertex in region.iter('Vertex'):
                    x, y = round(float(vertex.get('X')) / 16), round(float(vertex.get('Y')) / 16)
                    regions.append([x, y])
                cv2.drawContours(mask, np.array([regions]), -1, label, thickness=cv2.FILLED)
        info.pix_annotation = mask

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
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.patch_w_40x, info.patch_h_40x))
    patch = patch.resize(info.img_size)
    return patch

def otsu_threshold(info):
    print('perform otsu threshold..................')
    grayscale = filter_rgb_to_grayscale(info.slide_np)
    complement = filter_complement(grayscale)
    filtered = filter_otsu_threshold(complement)
    info.mask = np.clip(filtered, 0, 1)
    return

def get_tissue_rate(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    mask = info.mask[x:x+info.patch_w_1x,y:y+info.patch_h_1x]
    tissue_rate = int(np.sum(mask) * 100.0 / (mask.shape[0] * mask.shape[1]))
    return tissue_rate


def find_most_frequent_element(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    if len(unique_elements) == 1 and unique_elements[0] == 0:
        return 0
    max_count_index = np.argmax(counts)
    most_frequent_element = unique_elements[max_count_index]
    return most_frequent_element

def grid_wsi(info):
    print("grid and select patch from whole slide image................")
    info.num_row_patch = math.floor(info.h_1x / info.patch_h_1x)
    info.num_col_patch = math.floor(info.w_1x / info.patch_w_1x)
    print('num row:', info.num_row_patch, 'num col:', info.num_col_patch)
    for row_id in tqdm(range(0, info.num_row_patch)):
        for col_id in range(0, info.num_col_patch):
            tissue_rate = get_tissue_rate(row_id, col_id, info)
            if tissue_rate > info.threshold:
                patch = get_patch(row_id, col_id, info)
                if info.with_annotation:
                    label = info.pix_annotation[row_id * info.patch_w_1x:row_id * info.patch_w_1x + info.patch_w_1x,
                            col_id * info.patch_w_1x:col_id * info.patch_w_1x + info.patch_h_1x]
                    label = find_most_frequent_element(label)
                    patch.save(info.subject_dir + str(row_id) + '_' + str(col_id) + '_' + str(label) + '.png')
                else:
                    patch.save(info.subject_dir + str(row_id) + '_' + str(col_id) + '.png')

if __name__ == '__main__':
    info = bach_info()
    for svs_path in info.svs_path_all:
        name = os.path.splitext(os.path.split(svs_path)[-1])[0]
        print('preprocessing whole slide image:', name)
        xml_path = info.xml_root + name + '.xml'
        if not os.path.exists(xml_path):
            info.with_annotation = False
        else:
            info.with_annotation = True
        info.makedir(svs_path, xml_path)
        read_wsi(info)
        read_annotation(info)
        otsu_threshold(info)
        grid_wsi(info)