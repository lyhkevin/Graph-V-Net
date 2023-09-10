# Graph V-Net
We provide Pytorch implementations for our paper "A Hierarchical Graph V-Net with Semi-supervised Pre-training for Breast Cancer Histology Image Classification".
  
## 1. Introduction
Graph V-Net is a hierarchical graph convolutional network for patch-based breast cancer diagnosis. 
The proposed Graph V-Net classifies each patch within the whole slide image into four categories: normal, benign, carcinoma in situ, and invasive carcinoma. 

<p align="center">
    <img src="imgs/Overview.png"/> <br />
    <em>
    Figure 1. An overview of the proposed Graph V-Net.
    </em>
</p>

**Preview:**

Our proposed framework consists of two main components: 

- Pre-train the patch-level feature extractor with semi-supervised learning.

- Fine-tune the Graph V-Net in a supervised learning manner.

## 2. Graph V-Net Walkthrough

- ### Installation

  Install PyTorch and torchvision from http://pytorch.org and other dependencies. You can install all the dependencies by
  ```bash
  pip install -r ./requirements.txt
  ```
  
- ### Dataset Preparation

  Download the [BACH](https://zenodo.org/record/3632035) dataset, which including BACH training set (ICIAR2018_BACH_Challenge.zip) and BACH testing set (ICIAR2018_BACH_Challenge_TestDataset.zip). Unzip them in the `./dataset/` folder.
  Note that BACH dataset includes both microscopy images (ROI) and whole slide images, and we use whole slide images only. The folder structure should be like this:
  ```
  /dataset/
     ├── annotations
     ├── ICIAR2018_BACH_Challenge
     ├── ICIAR2018_BACH_Challenge_TestDataset
  ```
  We provide refined annotations for the BACH dataset in `./dataset/annotations/`. 
  <p align="center">
    <img src="imgs/annotation.png" width="50%"/> <br />
    <em>
    Figure 2. (a) The original annotation provided by BACH organizers. (b) A WSI relabeled by the pathologists from Yunnan Cancer Hospital. 
    </em>
</p>

- ### Date Preprocessing

  (1) After unzipping all the data, run the `./code/preprocessing/preprocessing_pretrain.py` first. This script will generate patches (from 31 labeled WSIs and 9 unlabeled WSIs) for pre-training in `./dataset/pretrain/`.  The file name of a specific patch indicates its spatial location and label. For instance, `./labeled/01/18_58_3/.png` indicates that the patch locates at the 18th row and 58th column of the WSI, and its label is 3 (invasive carcinoma).  
  (2) Then, run the `./code/preprocessing/preprocessing_finetune.py` to generate overlapping regions (from 31 labeled WSIs) for fine-tuning and testing in `./dataset/finetune/`. Each cropped region consists of 64 patches and the corresponding soft label is a numpy array with a size of (8,8,4), which indicates the average label of all pixels in the patch.

- ### Pre-training

  To pre-train the patch encoder, run `./code/pretrain/pretrain.py`. The weights of the patch encoder will be saved in `./weight/pretrain/`. We also provide the pre-trained weight in `./weight/pretrain/checkpoint.pth`. Run `./code/pretrain/visualize_attention.py` to visualize the attention maps of the patch encoder.

- ### Fine-tuning

   To fine-tune our framework, run `./code/train.py`. When fine-tuning is completed, the weights will be saved in `./weight/finetune/`. 

- ### Testing

  Run `test.py`, and the results (figure 3) will be saved in `./snapshot/test/`.

  <p align="center">
      <img src="imgs/test.png" width="100%"/> <br />
      <em>
      Figure 3. The prediction results of our framework.
      </em>
  </p>

[//]: # (## 3. Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # ()
[//]: # (```)

## 3. ToDo

- Upload our annotations for BACH dataset.

## 4. References
- ICIAR2018_BACH_Challenge: [[HERE]](https://www.sciencedirect.com/science/article/abs/pii/S1361841518307941)

- DINO: [[HERE]](https://github.com/facebookresearch/dino)

- Vision GNN: [[HERE]](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)

- python-wsi-preprocessing: [[HERE]](https://github.com/deroneriksson/python-wsi-preprocessing)