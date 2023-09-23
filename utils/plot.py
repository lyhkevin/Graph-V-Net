import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks as draw_mask

def drawing_mask(mask, slide):

    #color_others, color_benign, color_insitu, color_invasive, color_trash = (0,0,0), (255, 0, 255), (255, 51, 0), (255, 255, 0), (0, 221, 68)
    color_others, color_benign, color_insitu, color_invasive = (0, 0, 0), (0, 255, 0), (255, 255, 0), (255, 51, 0)
    Color = [color_others,color_benign, color_insitu,color_invasive]

    mask_bool = (np.arange(mask.max()) == mask[..., None] - 1).astype(bool)
    mask_tensor = torch.from_numpy(mask_bool)
    wsi_tensor = torch.from_numpy(slide)
    wsi_tensor = torch.permute(wsi_tensor, (2, 0, 1))
    mask_tensor = torch.permute(mask_tensor, (2, 0, 1))
    wsi_mask = draw_mask(wsi_tensor, mask_tensor, alpha=0.35, colors=Color[1:])
    wsi_mask = F.to_pil_image(wsi_mask)

    return wsi_mask


def drawing_annotation_mask(mask, slide):
    color_others, color_benign, color_insitu, color_invasive = (0, 0, 0), (0, 255, 0),(255, 255, 0), (255, 51, 0)
    Color = [color_others,color_benign, color_insitu,color_invasive]
    slide[:] = 0
    mask_bool = (np.arange(mask.max()) == mask[..., None] - 1).astype(bool)
    mask_tensor = torch.from_numpy(mask_bool)
    wsi_tensor = torch.from_numpy(slide)
    wsi_tensor = torch.permute(wsi_tensor, (2, 0, 1))
    mask_tensor = torch.permute(mask_tensor, (2, 0, 1))
    wsi_mask = draw_mask(wsi_tensor, mask_tensor, alpha=1, colors=Color[1:])
    wsi_mask = F.to_pil_image(wsi_mask)

    return wsi_mask
