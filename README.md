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

Our proposed framework consist of two main components: 

- Pre-train the patch-level feature extractor with semi-supervised learning.

- Fine-tune the Graph V-Net with the pre-trained patch-level feature extractor in a supervised learning manner.

## 2. Graph V-Net Walkthrough

- ### Installation

  Install PyTorch and torchvision from http://pytorch.org and other dependencies. You can install all the dependencies by
  ```bash
  pip install -r requirements.txt
  ```
  
- ### Dataset Preparation

  Download the [BACH](https://zenodo.org/record/3632035) dataset, which including BACH training set (ICIAR2018_BACH_Challenge.zip) and BACH testing set (ICIAR2018_BACH_Challenge_TestDataset.zip). Unzip them in the `./dataset/` folder.
  Note that BACH dataset includes both microscopy images (ROI) and whole slide images, and we use the whole slide images only. The folder structure should be like this:
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

  After preparing all the data, run the `./preprocessing/preprocessing_pretrain.py` first. This script will generate patches (from 31 labeled WSIs and 9 unlabeled WSIs) for pre-training in `./dataset/pretrain/`. 
  Then, run the `./preprocessing/preprocessing_finetune.py` to generate overlapping regions (from 31 labeled WSIs) for fine-tuning in `./dataset/finetune/`.

- ### Pre-training

  To pre-train the patch encoder, run `./code/semi/pretrain.py`. The weights will be saved in `./weight/pretrain/`.

- ### Fine-tuning

   To fine-tune our framework, run `./code/train.py`. You may change the default settings in the `./options/finetune_options.py`, especially the `data_rate` option to adjust the amount of paired data for fine-tuning. Besides, Besides, you can increase `num_workers` to speed up pre-training. The weights will be saved in `./weight/finetuned/`. Note that for MT-Net, the input size must be 256×256.

- ### Testing

  When fine-tuning is completed, the weights of the patch encoder and the Graph V-Net will be saved in `./code//weight/finetune/`. You can change the default settings in the `./options/test.py`. Then, run `test.py`, and the results will be saved in `./snapshot/test/`.

[//]: # (## 3. Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # ()
[//]: # (```)

## 3. References
- ICIAR2018_BACH_Challenge: [[HERE]](https://www.sciencedirect.com/science/article/abs/pii/S1361841518307941)

- DINO: [[HERE]](https://github.com/facebookresearch/dino)

- Vision GNN: [[HERE]](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)
